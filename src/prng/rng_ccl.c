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
#include <semaphore.h>
#include <assert.h>

/* Number of random number in buffer at each time.*/
#define NUMRN_DEFAULT 16777216

/* Number of iterations producing random numbers. */
#define NUMITER_DEFAULT 10000

/* Error handling macro. */
#define HANDLE_ERROR(err) \
	do { if ((err) != NULL) { \
		fprintf(stderr, "\nError at line %d: %s\n", __LINE__, (err)->message); \
		exit(EXIT_FAILURE); } \
	} while(0)

/* Kernels. */
#define KERNEL_INIT "init"
#define KERNEL_RNG "rng"
const char* kernel_filenames[] = { KERNEL_INIT ".cl", KERNEL_RNG ".cl" };

/* Thread semaphores. */
sem_t sem_rng;
sem_t sem_comm;

/* Information shared between main thread and data transfer/output thread. */
struct bufshare {

	/* Host buffer. */
	cl_ulong * bufhost;

	/* Device buffers. */
	CCLBuffer * bufdev1;
	CCLBuffer * bufdev2;

	/* Command queue for data transfers. */
	CCLQueue * cq;

	/* Possible transfer error. */
	CCLErr* err;

	/* Number of random numbers in buffer. */
	cl_uint numrn;

	/* Number of iterations producing random numbers. */
	unsigned int numiter;

	/* Buffer size in bytes. */
	size_t bufsize;

};

/* Write random numbers directly (as binary) to stdout. */
void * rng_out(void * arg) {

	/* Increment aux variable. */
	unsigned int i;

	/* Buffer pointers. */
	CCLBuffer * bufdev1, * bufdev2, * bufswp;

	/* Unwrap argument. */
	struct bufshare * bufs = (struct bufshare *) arg;

	/* Get initial buffers. */
	bufdev1 = bufs->bufdev1;
	bufdev2 = bufs->bufdev2;

	/* Read random numbers and write them to stdout. */
	for (i = 0; i < bufs->numiter; i++) {

		/* Wait for RNG kernel from previous iteration before proceding with
		 * next read. */
		sem_wait(&sem_rng);

		/* Read data from device buffer into host buffer. */
		ccl_buffer_enqueue_read(bufdev1, bufs->cq, CL_TRUE, 0,
			bufs->bufsize, bufs->bufhost, NULL, &bufs->err);

		/* Signal that read for current iteration is over. */
		sem_post(&sem_comm);

		/* If error occured in read, terminate thread and let main thread
		 * handle error. */
		if (bufs->err) return NULL;

		/* Write raw random numbers to stdout. */
		fwrite(bufs->bufhost, sizeof(cl_ulong), (size_t) bufs->numrn, stdout);
		fflush(stdout);

		/* Swap buffers. */
		bufswp = bufdev1;
		bufdev1 = bufdev2;
		bufdev2 = bufswp;

	}

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
	struct bufshare bufs = { NULL, NULL, NULL, NULL, NULL, 0, 0, 0 };

	/* Communications thread. */
	pthread_t comms_th;

	/* Increment aux variable. */
	unsigned int i;

	/* cf4ocl wrappers. */
	CCLContext * ctx = NULL;
	CCLDevice * dev = NULL;
	CCLProgram * prg = NULL;
	CCLKernel * kinit = NULL, *krng = NULL;
	CCLQueue * cq_main = NULL;
	CCLBuffer * bufdev1 = NULL, * bufdev2 = NULL, * bufswp = NULL;
	CCLEvent * evt_exec = NULL;

	/* Profiler object. */
	CCLProf* prof = NULL;

	/* Error management object. */
	CCLErr *err = NULL;

	/* Device name. */
	char* dev_name;

	/* Real and kernel work sizes. */
	size_t rws = 0, gws1 = 0, gws2 = 0, lws1 = 0, lws2 = 0;

	/* Program build log. */
	const char * bldlog;

	/* Initialize semaphores. */
	sem_init(&sem_rng, 0, 1);
	sem_init(&sem_comm, 0, 1);

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
		bufs.numiter = atoi(argv[2]);
	} else {
		/* No, use defaults. */
		bufs.numiter = NUMITER_DEFAULT;
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
	bufdev1 = ccl_buffer_new(
		ctx, CL_MEM_READ_WRITE, bufs.bufsize, NULL, &err);
	HANDLE_ERROR(err);
	bufdev2 = ccl_buffer_new(
		ctx, CL_MEM_READ_WRITE, bufs.bufsize, NULL, &err);
	HANDLE_ERROR(err);

	/* Pass reference of device buffers to shared struct. */
	bufs.bufdev1 = bufdev1;
	bufs.bufdev2 = bufdev2;

	/* Print information. */
	fprintf(stderr, "\n");
	fprintf(stderr, " * Device name                   : %s\n", dev_name);
	fprintf(stderr, " * Global/local work sizes (init): %u/%u\n",
		(unsigned int) gws1, (unsigned int) lws1);
	fprintf(stderr, " * Global/local work sizes (rng) : %u/%u\n",
		(unsigned int) gws2, (unsigned int) lws2);
	fprintf(stderr, " * Number of iterations          : %u\n",
		(unsigned int) bufs.numiter);

	/* Initialize profiling. */
	prof = ccl_prof_new();
	ccl_prof_start(prof);

	/* Invoke kernel for initializing random numbers. */
	evt_exec = ccl_kernel_set_args_and_enqueue_ndrange(kinit, cq_main, 1, NULL,
		(const size_t*) &gws1, (const size_t*) &lws1, NULL, &err,
		bufdev1, ccl_arg_priv(bufs.numrn, cl_uint), /* Kernel arguments. */
		NULL);
	HANDLE_ERROR(err);
	ccl_event_set_name(evt_exec, "INIT_KERNEL");

	/* Set fixed argument of RNG kernel (number of random numbers in buffer). */
	ccl_kernel_set_arg(krng, 0, ccl_arg_priv(bufs.numrn, cl_uint));

	/* Wait for initialization to finish. */
	ccl_queue_finish(cq_main, &err);
	HANDLE_ERROR(err);

	/* Invoke thread to output random numbers to stdout
	 * (in raw, binary form). */
	pthread_create(&comms_th, NULL, rng_out, &bufs);

	/* Produce random numbers. */
	for (i = 0; i < bufs.numiter - 1; i++) {

		/* Wait for read from previous iteration. */
		sem_wait(&sem_comm);

		/* Handle possible errors in comms thread. */
		HANDLE_ERROR(bufs.err);

		/* Run random number generation kernel. */
		evt_exec = ccl_kernel_set_args_and_enqueue_ndrange(krng, cq_main, 1,
			NULL, (const size_t*) &gws2, (const size_t*) &lws2, NULL, &err,
			ccl_arg_skip, bufdev1, bufdev2, /* Kernel arguments. */
			NULL);
		HANDLE_ERROR(err);
		ccl_event_set_name(evt_exec, "RNG_KERNEL");

		/* Wait for random number generation kernel to finish. */
		ccl_queue_finish(cq_main, &err);
		HANDLE_ERROR(err);

		/* Signal that RNG kernel from previous iteration is over. */
		sem_post(&sem_rng);

		/* Swap buffers. */
		bufswp = bufdev1;
		bufdev1 = bufdev2;
		bufdev2 = bufswp;

	}

	/* Wait for output thread to finish. */
	pthread_join(comms_th, NULL);

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
	if (bufdev1) ccl_buffer_destroy(bufdev1);
	if (bufdev2) ccl_buffer_destroy(bufdev2);
	if (cq_main) ccl_queue_destroy(cq_main);
	if (bufs.cq) ccl_queue_destroy(bufs.cq);
	if (prg) ccl_program_destroy(prg);
	if (ctx) ccl_context_destroy(ctx);

	/* Free host resources */
	if (bufs.bufhost) free(bufs.bufhost);

	/* Destroy semaphores. */
	sem_destroy(&sem_comm);
	sem_destroy(&sem_rng);

	/* Check that all cf4ocl wrapper objects are destroyed. */
	assert(ccl_wrapper_memcheck());

	/* Bye. */
	return EXIT_SUCCESS;

}
