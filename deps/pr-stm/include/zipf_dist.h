#ifndef ZIPF_DIST_H_GUARD
#define ZIPF_DIST_H_GUARD

#ifdef __cplusplus
extern "C" {
#endif
void zipf_setup(unsigned long nbItems, double param);
unsigned long zipf_gen();
#ifdef __cplusplus
}
#endif

#endif /* ZIPF_DIST_H_GUARD */
