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
#include <assert.h>
#include <sys/time.h>
#include <stdio.h>

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

/* Information shared between main thread and data transfer/output thread. */
struct bufshare {

	/* Host buffer. */
	cl_ulong * bufhost;

	/* Device buffer. */
	cl_mem bufdev;

	/* Command queue for data transfers. */
	cl_command_queue cq;

	/* Possible transfer error. */
	cl_int status;

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
	bufs->status = clFinish(bufs->cq);

	/* If error occurs let main thread handle it. */
	if (bufs->status != CL_SUCCESS) return NULL;

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

	/* Aux. variable for loops. */
	unsigned int i;

	/* Host buffer. */
	struct bufshare bufs = { NULL, NULL, NULL, 0, 0, 0 };

	/* Communications thread. */
	pthread_t comms_th;

	/* Thread status. */
	int sth;

	/* OpenCL objects. */
	cl_context ctx = NULL;
	cl_device_id dev = NULL;
	cl_program prg = NULL;
	cl_kernel kernels[2] = { NULL, NULL };
	cl_command_queue cq_main = NULL;
	cl_mem buf_main = NULL, bufswp = NULL;
	cl_event evt_kinit = NULL;
	cl_event * evts = NULL;
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

	/* Number of iterations producing random numbers. */
	unsigned int numiter;

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

	/* Get kernels. */
	status = clCreateKernelsInProgram(prg, 2, kernels, NULL);
	HANDLE_ERROR(status);

	/* Determine work sizes for each kernel. This is a minimum LOC approach
	 * which requires OpenCL >= 1.1. It does not account for the number of
	 * possibilities considered by the ccl_kernel_suggest_worksizes() cf4ocl
	 * function, namely multiple dimensions, OpenCL 1.0, kernel information
	 * unavailable, etc. */
	status = clGetKernelWorkGroupInfo(
		kernels[0], dev, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
		sizeof(size_t), &lws1, NULL);
	HANDLE_ERROR(status);
	gws1 = ((rws / lws1) + (((rws % lws1) > 0) ? 1 : 0)) * lws1;

	status = clGetKernelWorkGroupInfo(
		kernels[1], dev, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
		sizeof(size_t), &lws2, NULL);
	HANDLE_ERROR(status);
	gws2 = ((rws / lws2) + (((rws % lws2) > 0) ? 1 : 0)) * lws2;

	/* Allocate memory for host buffer. */
	bufs.bufhost = (cl_ulong *) malloc(bufs.bufsize);

	/* Create device buffers. */
	buf_main = clCreateBuffer(
		ctx, CL_MEM_READ_WRITE, bufs.bufsize, NULL, &status);
	HANDLE_ERROR(status);

	bufs.bufdev = clCreateBuffer(
		ctx, CL_MEM_READ_WRITE, bufs.bufsize, NULL, &status);
	HANDLE_ERROR(status);

	/* Initialize events buffers. */
	evts = (cl_event *) calloc(2 * numiter, sizeof(cl_event));

	/* Print information. */
	fprintf(stderr, "\n");
	fprintf(stderr, " * Device name                   : %s\n", dev_name);
	fprintf(stderr, " * Global/local work sizes (init): %u/%u\n",
		(unsigned int) gws1, (unsigned int) lws1);
	fprintf(stderr, " * Global/local work sizes (rng) : %u/%u\n",
		(unsigned int) gws2, (unsigned int) lws2);
	fprintf(stderr, " * Number of iterations          : %u\n",
		(unsigned int) numiter);

	/* Start profiling. */
	gettimeofday(&time0, NULL);

	/* Set arguments for initialization kernel. */
	status = clSetKernelArg(
		kernels[0], 0, sizeof(cl_mem), (const void *) &bufs.bufdev);
	HANDLE_ERROR(status);
	status = clSetKernelArg(
		kernels[0], 1, sizeof(cl_uint), (const void *) &bufs.numrn);
	HANDLE_ERROR(status);

	/* Invoke initialization kernel. */
	status = clEnqueueNDRangeKernel(cq_main, kernels[0], 1, NULL,
		(const size_t *) &gws1, (const size_t *) &lws1, 0, NULL, &evt_kinit);
	HANDLE_ERROR(status);

	/* Set fixed argument of RNG kernel (number of random numbers in buffer). */
	status = clSetKernelArg(
		kernels[1], 0, sizeof(cl_uint), (const void *) &bufs.numrn);
	HANDLE_ERROR(status);

	/* Wait for initialization to finish. */
	status = clFinish(cq_main);
	HANDLE_ERROR(status);

	/* Produce random numbers. */
	for (i = 0; i < numiter; i++) {

		/* Read data from device buffer into host buffer (non-blocking call). */
		status = clEnqueueReadBuffer(bufs.cq, bufs.bufdev, CL_FALSE, 0,
			bufs.bufsize, bufs.bufhost, 0, NULL, &evts[i * 2]);
		HANDLE_ERROR(status);

		/* Invoke thread to output random numbers to stdout
		 * (in raw, binary form). */
		sth = pthread_create(&comms_th, NULL, rng_out, &bufs);
		assert(sth == 0);

		/* Set RNG kernel arguments. */
		status = clSetKernelArg(
			kernels[1], 1, sizeof(cl_mem), (const void *) &bufs.bufdev);
		HANDLE_ERROR(status);

		status = clSetKernelArg(
			kernels[1], 2, sizeof(cl_mem), (const void *) &buf_main);
		HANDLE_ERROR(status);

		/* Run RNG kernel. */
		status = clEnqueueNDRangeKernel(cq_main, kernels[1], 1, NULL,
			(const size_t *) &gws2, (const size_t *) &lws2, 0, NULL,
			&evts[i * 2 + 1]);
		HANDLE_ERROR(status);

		/* Wait for transfer and for RNG kernel. */
		status = clWaitForEvents(2, (const cl_event *) &evts[i * 2]);
		HANDLE_ERROR(status);

		/* Wait for output thread to finish. */
		sth = pthread_join(comms_th, NULL);
		assert(sth == 0);
		HANDLE_ERROR(bufs.status);

		/* Swap buffers. */
		bufswp = buf_main;
		buf_main = bufs.bufdev;
		bufs.bufdev = bufswp;

	}

	/* Wait for all operations to finish. */
	status = clFinish(cq_main);
	HANDLE_ERROR(status);

	/* Stop profiling. */
	gettimeofday(&time1, NULL);

	/* Perform basic profiling calculations (i.e., we don't calculate overlaps,
	 * which automatically determined with the cf4ocl profiler). */

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

	/* Communication / RNG kernel time. */
	for (i = 0; i < numiter; i++) {

		/* Communication time. */
		status = clGetEventProfilingInfo(evts[i * 2],
			CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &tstart, NULL);
		HANDLE_ERROR(status);
		status = clGetEventProfilingInfo(evts[i * 2],
			CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &tend, NULL);
		HANDLE_ERROR(status);
		tcomms += tend - tstart;

		/* RNG kernel time. */
		status = clGetEventProfilingInfo(evts[i * 2 + 1],
			CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &tstart, NULL);
		HANDLE_ERROR(status);
		status = clGetEventProfilingInfo(evts[i * 2 + 1],
			CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &tend, NULL);
		HANDLE_ERROR(status);
		tkrng += tend - tstart;

	}

	/* Show basic profiling info. */
	fprintf(stderr, " * Total elapsed time                : %es\n", dt);
	fprintf(stderr, " * Total time in 'init' kernel       : %es\n",
		(double) tkinit);
	fprintf(stderr, " * Total time in 'rng' kernel        : %es\n",
		(double) tkrng);
	fprintf(stderr, " * Total time fetching data from GPU : %es\n",
		(double) tcomms);
	fprintf(stderr, "\n");

	/* Destroy OpenCL objects. */
	if (evt_kinit) clReleaseEvent(evt_kinit);
	for (i = 0; i < numiter * 2; i++) {
		if (evts[i]) clReleaseEvent(evts[i]);
	}
	if (buf_main) clReleaseMemObject(buf_main);
	if (bufs.bufdev) clReleaseMemObject(bufs.bufdev);
	if (cq_main) clReleaseCommandQueue(cq_main);
	if (bufs.cq) clReleaseCommandQueue(bufs.cq);
	if (kernels[0]) clReleaseKernel(kernels[0]);
	if (kernels[1]) clReleaseKernel(kernels[1]);
	if (prg) clReleaseProgram(prg);
	if (ctx) clReleaseContext(ctx);

	/* Free platforms buffer. */
	if (platfs) free(platfs);

	/* Free event buffers. */
	if (evts) free(evts);

	/* Free host resources */
	if (bufs.bufhost) free(bufs.bufhost);

	/* Free kernel sources. */
	if (ksources[0]) free(ksources[0]);
	if (ksources[1]) free(ksources[1]);

	/* Free device name. */
	free(dev_name);

	/* Bye. */
	return EXIT_SUCCESS;

}

