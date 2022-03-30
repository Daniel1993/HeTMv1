all : mat_create

mat_create : mat_create.o
	$(CC) -o $@ $^ -l bsd

%.o : %.c
	$(CC) -c -g $< -o $@

clean : 
	rm -rf mat_create *.o
