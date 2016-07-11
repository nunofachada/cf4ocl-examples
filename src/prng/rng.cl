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
 * Generates pseudo-random numbers from a buffer with seeds and saves them in
 * another buffer.
 */

__kernel void rng(
		const uint nseeds,
		__global ulong *in,
		__global ulong *out) {

	/* Global ID of current work-item. */
	size_t gid = get_global_id(0);

	/* Does this work-item has anything to do? */
	if (gid < nseeds) {

		/* Fetch current state. */
		ulong state = in[gid];

		/* Update state using simple xor-shift RNG. */
		state ^= (state << 21);
		state ^= (state >> 35);
		state ^= (state << 4);

		/* Save new state in buffer. */
		out[gid] = state;

	}
}


