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
 * Initialize a vector of random numbers in device.
 */

__kernel void init(
		__global uint2 *seeds,
		const uint nseeds) {

	/* Global ID of current work-item. */
	size_t gid = get_global_id(0);

	/* Does this work-item has anything to do? */
	if (gid < nseeds) {

		/* Final value. */
		uint2 final;

		/* Initialize low bits with global ID. */
		uint a = (uint) gid;

		/* Scramble seed using hashes described in
		 * http://www.burtleburtle.net/bob/hash/integer.html */

		/* Low bits. */
		a = (a + 0x7ed55d16) + (a << 12);
		a = (a ^ 0xc761c23c) ^ (a >> 19);
		a = (a + 0x165667b1) + (a << 5);
		a = (a + 0xd3a2646c) ^ (a << 9);
		a = (a + 0xfd7046c5) + (a << 3);
		a = (a ^ 0xb55a4f09) ^ (a >> 16);

		/* Keep low bits. */
		final.x = a;

		/* High bits. */
		a = (a ^ 61) ^ (a >> 16);
		a = a + (a << 3);
		a = a ^ (a >> 4);
		a = a * 0x27d4eb2d;
		a = a ^ (a >> 15);

		/* Keep high bits. */
		final.y = a;

		/* Save random number in buffer. */
		seeds[gid] = final;

	}
}


