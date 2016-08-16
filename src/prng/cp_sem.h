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
 * Header library for cross-platform use of semaphores. For Apple machines
 * GCD semaphores are used. Otherwise, POSIX semaphores are used. On Windows
 * this requires MinGW or Cygwin.
 */

#ifdef __APPLE__
#include <dispatch/dispatch.h>
#else
#include <semaphore.h>
#endif

/**
 * The semaphore object.
 * */
typedef struct {
#ifdef __APPLE__
	dispatch_semaphore_t sem;
#else
	sem_t sem;
#endif
} cp_sem_t;

/**
 * Initialize semaphore.
 * */
static inline void cp_sem_init(cp_sem_t * s, unsigned int val) {
#ifdef __APPLE__
	s->sem = dispatch_semaphore_create((long) val);
#else
	sem_init(&s->sem, 0, val);
#endif
}

/**
 * Destroy semaphore.
 * */
static inline void cp_sem_destroy(cp_sem_t * s) {
#ifdef __APPLE__
	dispatch_release((dispatch_object_t) s->sem);
#else
	sem_destroy(&s->sem);
#endif
}

/**
 * Wait on semaphore if value is zero, otherwise decrement semaphore.
 * */
static inline int cp_sem_wait(cp_sem_t * s) {
	int res;
#ifdef __APPLE__
	res = (int) dispatch_semaphore_wait(s->sem, DISPATCH_TIME_FOREVER);
#else
	res = sem_wait(&s->sem);
#endif
	return res;
}

/**
 * Unlock semaphore.
 * */
static inline void cp_sem_post(cp_sem_t * s) {
#ifdef __APPLE__
	dispatch_semaphore_signal(s->sem);
#else
	sem_post(&s->sem);
#endif
}
