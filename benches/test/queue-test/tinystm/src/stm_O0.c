/*
 * File:
 *   stm.c
 * Author(s):
 *   Pascal Felber <pascal.felber@unine.ch>
 *   Patrick Marlier <patrick.marlier@unine.ch>
 * Description:
 *   STM functions.
 *
 * Copyright (c) 2007-2014.
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation, version 2
 * of the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * This program has a dual license and can also be distributed
 * under the terms of the MIT license.
 */

#include <assert.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>

#include <pthread.h>
#include <sched.h>

#include "stm.h"
#include "stm_internal.h"

#include "utils.h"
#include "atomic.h"
#include "gc.h"

/*
 * Called by the CURRENT thread to commit a transaction.
 */
_CALLCONV int
stm_commit(void)
{
  TX_GET;
  return int_stm_commit(tx);
}

_CALLCONV int
stm_commit_tx(stm_tx_t *tx)
{
  return int_stm_commit(tx);
}

/*
 * Called by the CURRENT thread to abort a transaction.
 */
_CALLCONV void
stm_abort(int reason)
{
  TX_GET;
  stm_rollback(tx, reason | STM_ABORT_EXPLICIT);
}

_CALLCONV void
stm_abort_tx(stm_tx_t *tx, int reason)
{
  stm_rollback(tx, reason | STM_ABORT_EXPLICIT);
}

/*
 * Called by the CURRENT thread to load a word-sized value.
 */
_CALLCONV ALIGNED stm_word_t
stm_load(volatile stm_word_t *addr)
{
  TX_GET;
  return int_stm_load(tx, addr);
}

_CALLCONV stm_word_t
stm_load_tx(stm_tx_t *tx, volatile stm_word_t *addr)
{
  return int_stm_load(tx, addr);
}

/*
 * Called by the CURRENT thread to store a word-sized value.
 */
_CALLCONV ALIGNED void
stm_store(volatile stm_word_t *addr, stm_word_t value)
{
  TX_GET;
  int_stm_store(tx, addr, value);
}

_CALLCONV void
stm_store_tx(stm_tx_t *tx, volatile stm_word_t *addr, stm_word_t value)
{
  int_stm_store(tx, addr, value);
}

/*
 * Called by the CURRENT thread to store part of a word-sized value.
 */
_CALLCONV ALIGNED void
stm_store2(volatile stm_word_t *addr, stm_word_t value, stm_word_t mask)
{
  TX_GET;
  int_stm_store2(tx, addr, value, mask);
}

_CALLCONV void
stm_store2_tx(stm_tx_t *tx, volatile stm_word_t *addr, stm_word_t value, stm_word_t mask)
{
  int_stm_store2(tx, addr, value, mask);
}
