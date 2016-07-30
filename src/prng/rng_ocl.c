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
 * Generate random numbers with OpenCL using the OpenCL host API.
 *
 * Compile with gcc or clang:
 * $ gcc -pthread -Wall -std=c99 rng_ocl.c -o rng_ocl -lOpenCL
 */

#if defined(__APPLE__) || defined(__MACOSX)
	#include <OpenCL/opencl.h>
#else
	#include <CL/opencl.h>
#endif
#include <pthread.h>
#include <semaphore.h>
#include <assert.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>

/* Number of random number in buffer at each time.*/
#define NUMRN_DEFAULT 16777216

/* Number of iterations producing random numbers. */
#define NUMITER_DEFAULT 10000

/* Error handling macro. */
#define HANDLE_ERROR(status) \
	do { if (status != CL_SUCCESS) { \
		fprintf(stderr, "\nOpenCL error %d at line %d\n", status, __LINE__); \
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
	cl_mem bufdev1;
	cl_mem bufdev2;

	/* Command queue for data transfers. */
	cl_command_queue cq;

	/* Array of RNG kernel and memory transfer events. */
	cl_event * evts;

	/* Possible transfer error. */
	cl_int status;

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
	cl_mem bufdev1, bufdev2, bufswp;

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
		bufs->status = clEnqueueReadBuffer(bufs->cq, bufdev1, CL_TRUE, 0,
			bufs->bufsize, bufs->bufhost, 0, NULL, &bufs->evts[i * 2]);

		/* Signal that read for current iteration is over. */
		sem_post(&sem_comm);

		/* If error occurs let main thread handle it. */
		if (bufs->status != CL_SUCCESS) return NULL;

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

	/* Aux. variable for loops. */
	unsigned int i;

	/* Host buffer. */
	struct bufshare bufs = { NULL, NULL, NULL, NULL, NULL, 0, 0, 0, 0 };

	/* Communications thread. */
	pthread_t comms_th;

	/* OpenCL objects. */
	cl_context ctx = NULL;
	cl_device_id dev = NULL;
	cl_program prg = NULL;
	cl_kernel kinit = NULL, krng = NULL;
	cl_command_queue cq_main = NULL;
	cl_mem bufdev1 = NULL, bufdev2 = NULL, bufswp = NULL;
	cl_event evt_kinit = NULL;
	cl_platform_id * platfs = NULL;

	/* Context properties. */
	cl_context_properties ctx_prop[3] = { CL_CONTEXT_PLATFORM, 0, 0 };

 	/* Number of platforms. */
 	cl_uint nplatfs;

 	/* Number of devices in platform. */
 	cl_uint ndevs;

	/* Device name. */
	char* dev_name;

	/* Status flag. */
	cl_int status;

	/* Size of information returned by clGet*Info functions. */
	size_t infosize;

	/* Generic vector where to put information returned by clGet*Info
	 * functions. */
	void * info = NULL;

	/* Real and kernel work sizes. */
	size_t rws, gws1, gws2, lws1, lws2;

	/* File pointer for files containing kernels. */
	FILE * fp;

	/* Length of kernel source code. */
	size_t klens[2];

	/* Kernel sources. */
	char * ksources[2] = { NULL, NULL };

	/* Variables for measuring execution time. */
	struct timeval time1, time0;
	double dt = 0;
	cl_ulong tstart, tend, tkinit = 0, tcomms = 0, tkrng = 0;

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

	/* Determine number of OpenCL platforms. */
	status = clGetPlatformIDs(0, NULL, &nplatfs);
	HANDLE_ERROR(status);

	/* Allocate memory for existing platforms. */
	platfs = (cl_platform_id*) malloc(sizeof(cl_platform_id) * nplatfs);

	/* Get existing OpenCL platforms. */
	status = clGetPlatformIDs(nplatfs, platfs, NULL);
	HANDLE_ERROR(status);

	/* Cycle through platforms until a GPU device is found. */
	for (i = 0; i < nplatfs; i++) {

		/* Determine number of GPU devices in current platform. */
		status = clGetDeviceIDs(platfs[i], CL_DEVICE_TYPE_GPU, 0, NULL, &ndevs);
		if (status == CL_DEVICE_NOT_FOUND) continue;
		else HANDLE_ERROR(status);

		/* Was any GPU device found in current platform? */
		if (ndevs > 0) {

			/* If so, get first device. */
			status = clGetDeviceIDs(
				platfs[i], CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
			HANDLE_ERROR(status);

			/* Set the current platform as a context property. */
			ctx_prop[1] = (cl_context_properties) platfs[i];

			/* No need to cycle any more platforms. */
			break;
		}
	}

	/* If no GPU device was found, give up. */
	assert(dev != NULL);

	/* Get device name. */
	status = clGetDeviceInfo(dev, CL_DEVICE_NAME, 0, NULL, &infosize);
	HANDLE_ERROR(status);
	dev_name = (char *) malloc(infosize);
	status = clGetDeviceInfo(
		dev, CL_DEVICE_NAME, infosize, (void *) dev_name, NULL);
	HANDLE_ERROR(status);

	/* Create context. */
	ctx = clCreateContext(ctx_prop, 1, &dev, NULL, NULL, &status);
	HANDLE_ERROR(status);

	/* Create command queues. Here we use the "old" queue constructor, which is
	 * deprecated in OpenCL >= 2.0, and may throw a warning if compiled against
	 * such OpenCL versions. In cf4ocl the appropriate constructor is invoked
	 * depending on the underlying platform and the OpenCL version cf4ocl was
	 * built against. */
	cq_main = clCreateCommandQueue(
		ctx, dev, CL_QUEUE_PROFILING_ENABLE, &status);
	HANDLE_ERROR(status);

	bufs.cq = clCreateCommandQueue(
		ctx, dev, CL_QUEUE_PROFILING_ENABLE, &status);
	HANDLE_ERROR(status);

	/* Read kernel sources into strings. */
	for (i = 0; i < 2; i++) {

		fp = fopen (kernel_filenames[i], "rb");
		assert(fp != NULL);
		fseek (fp, 0, SEEK_END);
		klens[i] = (size_t) ftell(fp);
		fseek(fp, 0, SEEK_SET);
		ksources[i] = malloc(klens[i]);
		fread(ksources[i], 1, klens[i], fp);
		fclose (fp);
	}

	/* Create program. */
	prg = clCreateProgramWithSource(ctx, 2, (const char **) ksources,
		(const size_t *) klens, &status);
	HANDLE_ERROR(status);

	/* Build program. */
	status = clBuildProgram(
		prg, 1, (const cl_device_id *) &dev, NULL, NULL, NULL);

	/* Print build log in case of error. */
	if (status == CL_BUILD_PROGRAM_FAILURE) {

		/* Get size of build log. */
		status = clGetProgramBuildInfo(
			prg, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &infosize);
		HANDLE_ERROR(status);

		/* Allocate space for build log. */
		info = malloc(infosize);

		/* Get build log. */
		status = clGetProgramBuildInfo(
			prg, dev, CL_PROGRAM_BUILD_LOG, infosize, info, NULL);
		HANDLE_ERROR(status);

		/* Show build log. */
		fprintf(stderr, "Error building program: \n%s", (char *) info);

		/* Release build log. */
		free(info);

		/* Stop program. */
		exit(EXIT_FAILURE);

	} else {

		HANDLE_ERROR(status);

	}

	/* Create init kernel. */
	kinit = clCreateKernel(prg, KERNEL_INIT, &status);
	HANDLE_ERROR(status);

	/* Create rng kernel. */
	krng = clCreateKernel(prg, KERNEL_RNG, &status);
	HANDLE_ERROR(status);

	/* Determine work sizes for each kernel. This is a minimum LOC approach
	 * which requires OpenCL >= 1.1. It does not account for the number of
	 * possibilities considered by the ccl_kernel_suggest_worksizes() cf4ocl
	 * function, namely multiple dimensions, OpenCL 1.0, kernel information
	 * unavailable, etc. */
	status = clGetKernelWorkGroupInfo(
		kinit, dev, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
		sizeof(size_t), &lws1, NULL);
	HANDLE_ERROR(status);
	gws1 = ((rws / lws1) + (((rws % lws1) > 0) ? 1 : 0)) * lws1;

	status = clGetKernelWorkGroupInfo(
		krng, dev, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
		sizeof(size_t), &lws2, NULL);
	HANDLE_ERROR(status);
	gws2 = ((rws / lws2) + (((rws % lws2) > 0) ? 1 : 0)) * lws2;

	/* Allocate memory for host buffer. */
	bufs.bufhost = (cl_ulong *) malloc(bufs.bufsize);

	/* Create device buffers. */
	bufdev1 = clCreateBuffer(
		ctx, CL_MEM_READ_WRITE, bufs.bufsize, NULL, &status);
	HANDLE_ERROR(status);

	bufdev2 = clCreateBuffer(
		ctx, CL_MEM_READ_WRITE, bufs.bufsize, NULL, &status);
	HANDLE_ERROR(status);

	/* Pass reference of device buffers to shared struct. */
	bufs.bufdev1 = bufdev1;
	bufs.bufdev2 = bufdev2;

	/* Initialize events buffers. */
	bufs.evts = (cl_event *) calloc(2 * bufs.numiter - 1, sizeof(cl_event));

	/* Print information. */
	fprintf(stderr, "\n");
	fprintf(stderr, " * Device name                   : %s\n", dev_name);
	fprintf(stderr, " * Global/local work sizes (init): %u/%u\n",
		(unsigned int) gws1, (unsigned int) lws1);
	fprintf(stderr, " * Global/local work sizes (rng) : %u/%u\n",
		(unsigned int) gws2, (unsigned int) lws2);
	fprintf(stderr, " * Number of iterations          : %u\n",
		(unsigned int) bufs.numiter);

	/* Start profiling. */
	gettimeofday(&time0, NULL);

	/* Set arguments for initialization kernel. */
	status = clSetKernelArg(
		kinit, 0, sizeof(cl_mem), (const void *) &bufdev1);
	HANDLE_ERROR(status);
	status = clSetKernelArg(
		kinit, 1, sizeof(cl_uint), (const void *) &bufs.numrn);
	HANDLE_ERROR(status);

	/* Invoke initialization kernel. */
	status = clEnqueueNDRangeKernel(cq_main, kinit, 1, NULL,
		(const size_t *) &gws1, (const size_t *) &lws1, 0, NULL, &evt_kinit);
	HANDLE_ERROR(status);

	/* Set fixed argument of RNG kernel (number of random numbers in buffer). */
	status = clSetKernelArg(
		krng, 0, sizeof(cl_uint), (const void *) &bufs.numrn);
	HANDLE_ERROR(status);

	/* Wait for initialization to finish. */
	status = clFinish(cq_main);
	HANDLE_ERROR(status);

	/* Invoke thread to output random numbers to stdout
	 * (in raw, binary form). */
	pthread_create(&comms_th, NULL, rng_out, &bufs);

	/* Produce random numbers. */
	for (i = 0; i < bufs.numiter - 1; i++) {

		/* Set RNG kernel arguments. */
		status = clSetKernelArg(
			krng, 1, sizeof(cl_mem), (const void *) &bufdev1);
		HANDLE_ERROR(status);

		status = clSetKernelArg(
			krng, 2, sizeof(cl_mem), (const void *) &bufdev2);
		HANDLE_ERROR(status);

		/* Wait for read from previous iteration. */
		sem_wait(&sem_comm);

		/* Handle possible errors in comms thread. */
		HANDLE_ERROR(bufs.status);

		/* Run RNG kernel. */
		status = clEnqueueNDRangeKernel(cq_main, krng, 1, NULL,
			(const size_t *) &gws2, (const size_t *) &lws2, 0, NULL,
			&bufs.evts[i * 2 + 1]);
		HANDLE_ERROR(status);

		/* Wait for random number generation kernel to finish. */
		status = clFinish(cq_main);
		HANDLE_ERROR(status);

		/* Signal that RNG kernel from previous iteration is over. */
		sem_post(&sem_rng);

		/* Swap buffers. */
		bufswp = bufdev1;
		bufdev1 = bufdev2;
		bufdev2 = bufswp;

	}

	/* Wait for output thread to finish. */
	pthread_join(comms_th, NULL);

	/* Stop profiling. */
	gettimeofday(&time1, NULL);

	/* Perform basic profiling calculations (i.e., we don't calculate overlaps,
	 * which are automatically determined with the cf4ocl profiler). */

	/* Total time. */
	dt = time1.tv_sec - time0.tv_sec;
	if (time1.tv_usec >= time0.tv_usec)
		dt = dt + (time1.tv_usec - time0.tv_usec) * 1e-6;
	else
		dt = (dt-1) + (1e6 + time1.tv_usec - time0.tv_usec) * 1e-6;

	/* Initialization kernel time. */
	status = clGetEventProfilingInfo(evt_kinit, CL_PROFILING_COMMAND_START,
		sizeof(cl_ulong), &tstart, NULL);
	HANDLE_ERROR(status);
	status = clGetEventProfilingInfo(evt_kinit, CL_PROFILING_COMMAND_END,
		sizeof(cl_ulong), &tend, NULL);
	tkinit = tend - tstart;

	/* Communication time. */
	for (i = 0; i < bufs.numiter; i++) {

		status = clGetEventProfilingInfo(bufs.evts[i * 2],
			CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &tstart, NULL);
		HANDLE_ERROR(status);
		status = clGetEventProfilingInfo(bufs.evts[i * 2],
			CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &tend, NULL);
		HANDLE_ERROR(status);
		tcomms += tend - tstart;

	}

	/* RNG kernel time. */
	for (i = 0; i < bufs.numiter - 1; i++) {

		/* RNG kernel time. */
		status = clGetEventProfilingInfo(bufs.evts[i * 2 + 1],
			CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &tstart, NULL);
		HANDLE_ERROR(status);
		status = clGetEventProfilingInfo(bufs.evts[i * 2 + 1],
			CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &tend, NULL);
		HANDLE_ERROR(status);
		tkrng += tend - tstart;

	}

	/* Show basic profiling info. */
	fprintf(stderr, " * Total elapsed time                : %es\n", dt);
	fprintf(stderr, " * Total time in 'init' kernel       : %es\n",
		(double) (tkinit * 1e-9));
	fprintf(stderr, " * Total time in 'rng' kernel        : %es\n",
		(double) (tkrng * 1e-9));
	fprintf(stderr, " * Total time fetching data from GPU : %es\n",
		(double) (tcomms * 1e-9));
	fprintf(stderr, "\n");

	/* Destroy OpenCL objects. */
	if (evt_kinit) clReleaseEvent(evt_kinit);
	for (i = 0; i < bufs.numiter * 2 - 1; i++) {
		if (bufs.evts[i]) clReleaseEvent(bufs.evts[i]);
	}
	if (bufdev1) clReleaseMemObject(bufdev1);
	if (bufdev2) clReleaseMemObject(bufdev2);
	if (cq_main) clReleaseCommandQueue(cq_main);
	if (bufs.cq) clReleaseCommandQueue(bufs.cq);
	if (kinit) clReleaseKernel(kinit);
	if (krng) clReleaseKernel(krng);
	if (prg) clReleaseProgram(prg);
	if (ctx) clReleaseContext(ctx);

	/* Free platforms buffer. */
	if (platfs) free(platfs);

	/* Free event buffers. */
	if (bufs.evts) free(bufs.evts);

	/* Free host resources */
	if (bufs.bufhost) free(bufs.bufhost);

	/* Free device name. */
	if (dev_name) free(dev_name);

	/* Bye. */
	return EXIT_SUCCESS;

}

