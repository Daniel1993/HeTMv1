#ifndef BARRIER_H_GUARD_
#define BARRIER_H_GUARD_

typedef struct ticket_barrier_ ticket_barrier_t;
struct ticket_barrier_
{
	volatile unsigned count_next;
	volatile unsigned total;
	volatile int is_stop;
	union
	{
		struct
		{
			volatile unsigned count_in;
			volatile unsigned count_out;
		};
		volatile unsigned long long reset;
	};
};

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

void ticket_barrier_init(ticket_barrier_t *b, unsigned count);
void ticket_barrier_destroy(ticket_barrier_t *b);
int ticket_barrier_cross(ticket_barrier_t *b);
int ticket_barrier_reset(ticket_barrier_t *b);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* BARRIER_H_GUARD_ */
