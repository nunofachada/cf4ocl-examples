/*
 * This file is part of cf4ocl-examples.
 * Copyright (C) 2016 Nuno Fachada
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 * */

/**
 * @file
 * Generate random numbers with OpenCL using the cf4ocl library.
 *
 * Compile with gcc or clang:
 * $ gcc -pthread -Wall -std=c99 `pkg-config --cflags cf4ocl2` \
 *       rng_ccl.c -o rng_ccl `pkg-config --libs cf4ocl2`
 */

#include <cf4ocl2.h>
#include <pthread.h>
#include <assert.h>

/* Number of random number in buffer at each time.*/
#define NUMRN_DEFAULT 16777216

/* Number of iterations producing random numbers. */
#define NUMITER_DEFAULT 10000

/* Error handling macro. */
#define HANDLE_ERROR(err) \
	do { if (err != NULL) { \
		fprintf(stderr, "\nError at line %d: %s\n", __LINE__, err->message); \
		exit(EXIT_FAILURE); } \
	} while(0)

/* Kernels. */
#define KERNEL_INIT "init"
#define KERNEL_RNG "rng"
const char* kernel_filenames[] = { KERNEL_INIT ".cl", KERNEL_RNG ".cl" };

/* Information shared between main thread and data transfer/output thread. */
struct bufshare {

	/* Host buffer. */
	cl_ulong * bufhost;

	/* Device buffer. */
	CCLBuffer * bufdev;

	/* Command queue for data transfers. */
	CCLQueue * cq;

	/* Possible transfer error. */
	CCLErr* err;

	/* Number of random numbers in buffer. */
	cl_uint numrn;

	/* Buffer size in bytes. */
	size_t bufsize;

};

/* Write random numbers directly (as binary) to stdout. */
void * rng_out(void * arg) {

	/* Unwrap argument. */
	struct bufshare * bufs = (struct bufshare *) arg;

	/* Wait on transfer. */
	ccl_queue_finish(bufs->cq, &bufs->err);

	/* If error occurs let main thread handle it. */
	if (bufs->err) return NULL;

	/* Write*/
	fwrite(bufs->bufhost, sizeof(cl_ulong), (size_t) bufs->numrn, stdout);

	/* Bye. */
	return NULL;
}

/**
 * Main program.
 *
 * @param argc Number of command line arguments.
 * @param argv Vector of command line arguments.
 * @return `EXIT_SUCCESS` if program terminates successfully, or another
 * `EXIT_FAILURE` if an error occurs.
 * */
int main(int argc, char **argv) {

	/* Host buffer. */
	struct bufshare bufs = { NULL, NULL, NULL, NULL, 0, 0 };

	/* Communications thread. */
	pthread_t comms_th;

	/* Thread status. */
	int sth;

	/* cf4ocl wrappers. */
	CCLContext * ctx = NULL;
	CCLDevice * dev = NULL;
	CCLProgram * prg = NULL;
	CCLKernel * kinit = NULL, *krng = NULL;
	CCLQueue * cq_main = NULL;
	CCLBuffer * buf_main = NULL, * bufswp = NULL;
	CCLEvent * evt_exec = NULL, * evt_comms = NULL;
	CCLEventWaitList ewl = NULL;

	/* Profiler object. */
	CCLProf* prof = NULL;

	/* Error management object. */
	CCLErr *err = NULL;

	/* Device name. */
	char* dev_name;

	/* Real and kernel work sizes. */
	size_t rws, gws1, gws2, lws1, lws2;

	/* Number of iterations producing random numbers. */
	unsigned int numiter;

	/* Program build log. */
	const char * bldlog;

	/* Did user specify a number of random numbers? */
	if (argc >= 2) {
		/* Yes, use it. */
		bufs.numrn = atoi(argv[1]);
		bufs.bufsize = bufs.numrn * sizeof(cl_ulong);
	} else {
		/* No, use defaults. */
		bufs.numrn = NUMRN_DEFAULT;
		bufs.bufsize = NUMRN_DEFAULT * sizeof(cl_ulong);
	}
	rws = (size_t) bufs.numrn;

	/* Did user specify a number of iterations producing random numbers? */
	if (argc >= 3) {
		/* Yes, use it. */
		numiter = atoi(argv[2]);
	} else {
		/* No, use defaults. */
		numiter = NUMITER_DEFAULT;
	}

	/* Setup OpenCL context with GPU device. */
	ctx = ccl_context_new_gpu(&err);
	HANDLE_ERROR(err);

	/* Get device. */
	dev = ccl_context_get_device(ctx, 0, &err);
	HANDLE_ERROR(err);

	/* Get device name. */
	dev_name = ccl_device_get_info_array(dev, CL_DEVICE_NAME, char*, &err);
	HANDLE_ERROR(err);

	/* Create command queues. */
	cq_main = ccl_queue_new(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &err);
	HANDLE_ERROR(err);
	bufs.cq = ccl_queue_new(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &err);
	HANDLE_ERROR(err);

	/* Create program. */
	prg = ccl_program_new_from_source_files(ctx, 2, kernel_filenames, &err);
	HANDLE_ERROR(err);

	/* Build program. */
	ccl_program_build(prg, NULL, &err);

	/* Print build log in case of error. */
	if ((err) && (err->code == CL_BUILD_PROGRAM_FAILURE)) {
		bldlog = ccl_program_get_build_log(prg, &err);
		HANDLE_ERROR(err);
		fprintf(stderr, "Error building program: \n%s", bldlog);
		exit(EXIT_FAILURE);
	}
	HANDLE_ERROR(err);

	/* Get kernels. */
	kinit = ccl_program_get_kernel(prg, KERNEL_INIT, &err);
	HANDLE_ERROR(err);
	krng = ccl_program_get_kernel(prg, KERNEL_RNG, &err);
	HANDLE_ERROR(err);

	/* Determine preferred work sizes for each kernel. */
	ccl_kernel_suggest_worksizes(kinit, dev, 1, &rws, &gws1, &lws1, &err);
	HANDLE_ERROR(err);
	ccl_kernel_suggest_worksizes(krng, dev, 1, &rws, &gws2, &lws2, &err);
	HANDLE_ERROR(err);

	/* Allocate memory for host buffer. */
	bufs.bufhost = (cl_ulong*) malloc(bufs.bufsize);

	/* Create device buffers. */
	buf_main = ccl_buffer_new(
		ctx, CL_MEM_READ_WRITE, bufs.bufsize, NULL, &err);
	HANDLE_ERROR(err);
	bufs.bufdev = ccl_buffer_new(
		ctx, CL_MEM_READ_WRITE, bufs.bufsize, NULL, &err);
	HANDLE_ERROR(err);

	/* Print information. */
	fprintf(stderr, "\n");
	fprintf(stderr, " * Device name                   : %s\n", dev_name);
	fprintf(stderr, " * Global/local work sizes (init): %u/%u\n",
		(unsigned int) gws1, (unsigned int) lws1);
	fprintf(stderr, " * Global/local work sizes (rng) : %u/%u\n",
		(unsigned int) gws2, (unsigned int) lws2);
	fprintf(stderr, " * Number of iterations          : %u\n",
		(unsigned int) numiter);

	/* Initialize profiling. */
	prof = ccl_prof_new();
	ccl_prof_start(prof);

	/* Invoke kernel for initializing random numbers. */
	evt_exec = ccl_kernel_set_args_and_enqueue_ndrange(kinit, cq_main, 1, NULL,
		(const size_t*) &gws1, (const size_t*) &lws1, NULL, &err,
		bufs.bufdev, ccl_arg_priv(bufs.numrn, cl_uint), /* Kernel arguments. */
		NULL);
	HANDLE_ERROR(err);
	ccl_event_set_name(evt_exec, "INIT_KERNEL");

	/* Set fixed argument of RNG kernel (number of random numbers in buffer). */
	ccl_kernel_set_arg(krng, 0, ccl_arg_priv(bufs.numrn, cl_uint));

	/* Wait for initialization to finish. */
	ccl_queue_finish(cq_main, &err);
	HANDLE_ERROR(err);

	/* Produce random numbers. */
	for (unsigned int i = 0; i < numiter; i++) {

		/* Read data from device buffer into host buffer (non-blocking call). */
		evt_comms = ccl_buffer_enqueue_read(bufs.bufdev, bufs.cq, CL_FALSE, 0,
			bufs.bufsize, bufs.bufhost, NULL, &err);
		HANDLE_ERROR(err);

		/* Invoke thread to output random numbers to stdout
		 * (in raw, binary form). */
		sth = pthread_create(&comms_th, NULL, rng_out, &bufs);
		assert(sth == 0);

		/* Run random number generation kernel. */
		evt_exec = ccl_kernel_set_args_and_enqueue_ndrange(krng, cq_main, 1,
			NULL, (const size_t*) &gws2, (const size_t*) &lws2, NULL, &err,
			ccl_arg_skip, bufs.bufdev, buf_main, /* Kernel arguments. */
			NULL);
		HANDLE_ERROR(err);
		ccl_event_set_name(evt_exec, "RNG_KERNEL");

		/* Wait for transfer and for RNG kernel. */
		ccl_event_wait(ccl_ewl(&ewl, evt_comms, evt_exec, NULL), &err);
		HANDLE_ERROR(err);

		/* Wait for output thread to finish. */
		sth = pthread_join(comms_th, NULL);
		assert(sth == 0);
		HANDLE_ERROR(bufs.err);

		/* Swap buffers. */
		bufswp = buf_main;
		buf_main = bufs.bufdev;
		bufs.bufdev = bufswp;

	}

	/* Wait for all operations to finish. */
	ccl_queue_finish(cq_main, &err);
	HANDLE_ERROR(err);

	/* Perform profiling. */
	ccl_prof_stop(prof);
	ccl_prof_add_queue(prof, "Main", cq_main);
	ccl_prof_add_queue(prof, "Comms", bufs.cq);
	ccl_prof_calc(prof, &err);
	HANDLE_ERROR(err);

	/* Show profiling info. */
	fprintf(stderr, "%s",
		ccl_prof_get_summary(prof,
			CCL_PROF_AGG_SORT_TIME, CCL_PROF_OVERLAP_SORT_DURATION));

	/* Destroy profiler object. */
	ccl_prof_destroy(prof);

	/* Destroy cf4ocl wrappers - only the ones created with ccl_*_new()
	 * functions. */
	if (buf_main) ccl_buffer_destroy(buf_main);
	if (bufs.bufdev) ccl_buffer_destroy(bufs.bufdev);
	if (cq_main) ccl_queue_destroy(cq_main);
	if (bufs.cq) ccl_queue_destroy(bufs.cq);
	if (kinit) ccl_kernel_destroy(kinit);
	if (krng) ccl_kernel_destroy(krng);
	if (prg) ccl_program_destroy(prg);
	if (ctx) ccl_context_destroy(ctx);

	/* Free host resources */
	if (bufs.bufhost) free(bufs.bufhost);

	/* Bye. */
	return EXIT_SUCCESS;

}
