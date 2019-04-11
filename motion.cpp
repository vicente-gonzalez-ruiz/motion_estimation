/* Motion Vector Fields (MVF) stuff */

#define X_FIELD  0 /* X components of a field */
#define Y_FIELD  1 /* Y components of a field */

#define MVC_TYPE short  /* MVC = Motion Vectors Components */

/* Limits */
#define PIXELS_IN_X_MAX 16384

template <typename TYPE>       
class motion {

public:

  /* Allocates a bidirectional motion vector field */
  TYPE ***alloc(int y_dim, int x_dim) {
    TYPE ***data = new TYPE ** [ 2 ];
    for(int f=0; f<2; f++) {
	data[f] = new TYPE * [y_dim];
	for(int y=0; y<y_dim; y++) {
	  data[f][y] = new TYPE [x_dim];
	}
      }
    }
    return data;
  }

  /* Deallocates a MVF */
  void free(TYPE ***data, int y_dim) {
    for(int f=0; f<2; f++) {
      for(int y=0; y<y_dim; y++) {
	delete data[f][y];
      }
      delete data[f];
    }
    delete data;
  }

  /* Reads from disk a MVF */
  void read_(FILE *fd, TYPE ***data, int y_dim, int x_dim) {
    char x[80];
    fgets(x, 80, fd); /* Magic number */
    fgets(x, 80, fd); /* rows and cols */
    fgets(x, 80, fd); /* Max value */
    for(int f=0; f<2; f++) {
      for(int y=0; y<y_dim; y++) {
	int read = fread(data[f][y], x_dim, sizeof(TYPE), fd);
      }
    }
  }

  /* Writes to disk a MVF */
  void write_(FILE *fd, TYPE ***data, int y_dim, int x_dim) {
    fprintf(fd, "P5\n");
    fprintf(fd, "%d %d\n", x_dim, y_dim);
    fprintf(fd, "65535\n");
    for(int f=0; f<2; f++) {
      for(int y=0; y<y_dim; y++) {
	fwrite(data[f][y], x_dim, sizeof(TYPE), fd);
      }
    }
  }

  
  void read_field(TYPE ***data,
		  int blocks_in_y,
		  int blocks_in_x,
		  char *fn,
		  int field
#if defined (__WARNING__) || defined (__INFO__) || defined (__DEBUG__)
		      ,
		      char *msg
#endif /* __INFO__ */
		      ) {
    char fn_[80];
    sprintf(fn_, "%s/%04d.pgm", fn, field);
    FILE *fd  = fopen(fn_, "r");
    if(!fd) {
#if defined (__WARNING__)
      warning("%s: using \"/dev/zero\" instead of \"%s\"\n", msg, fn_);
#endif /* __WARNING__ */
      fd = fopen("/dev/zero", "r");
    }
    motion::read_(fd, data, blocks_in_y, blocks_in_x);
#if defined (__INFO__)
    info("%s: read %dx%d from \"%s\"\n", msg, blocks_in_y, blocks_in_x, fn_);
#endif /* __INFO__ */

    fclose(fd);
  }

  void write_field(TYPE ****data,
		   int blocks_in_y,
		   int blocks_in_x,
		   char *fn,
		   int field
#if defined (__DEBUG__) || defined (__INFO__) || defined (__WARNING__)
		   ,
		   char *msg
#endif /* __INFO__ */
		   ) {
    char fn_[80];
    sprintf(fn_, "%s/%04d.rawl", fn, field);
    FILE *fd  = fopen(fn_, "w");
#if defined (__DEBUG__)
    if(!fd) {
      error("%s: unable to create the file \"%s\" ... aborting!\n", msg, fn_);
      abort();
    }
#endif /* __DEBUG__ */
    motion::write_(fd, data, blocks_in_y, blocks_in_x);
#if defined (__INFO__)
    info("%s: written %dx%d to \"%s\"\n", msg, blocks_in_y, blocks_in_x, fn_);
#endif /* __INFO__ */
    fclose(fd);
  }

  ////////////////////////////////////////////////////////////////
  
#ifdef _1_
  void read(FILE *fd, TYPE **data, int y_dim, int x_dim) {
    char x[80];
    fgets(x, 80, fd); /* Magic number */
    fgets(x, 80, fd); /* rows and cols */
    fgets(x, 80, fd); /* Max value */
    for(int y=0; y<y_dim; y++) {
      int read = fread(data[y], x_dim, sizeof(TYPE), fd);
#ifdef _1_
#if defined (__INFO__) /** Sign and magnitude */
      for(int x=0; x<x_dim; x++) {
	info("%2d ", data[y][x]);
      }
      info("\n");
#endif /* __INFO__ */
#endif
    }
  }

  void write(FILE *fd, TYPE **data, int y_dim, int x_dim) {
    fprintf(fd, "P5\n");
    fprintf(fd, "%d %d\n", x_dim, y_dim);
    fprintf(fd, "65535\n");
    for(int y=0; y<y_dim; y++) {
      fwrite(data[y], x_dim, sizeof(TYPE), fd);
#ifdef _1_
#if defined (__INFO__) /** Sign and magnitude */
      for(int x=0; x<x_dim; x++) {
	info("%2d ", data[y][x]);
      }
      info("\n");
#endif /* __INFO__ */
#endif
    }
  }
#endif

#ifdef _1_
  void read_component(TYPE **data,
		      int blocks_in_y,
		      int blocks_in_x,
		      char *fn,
		      int field,
		      int component
#if defined (__WARNING__) || defined (__INFO__) || defined (__DEBUG__)
		      ,
		      char *msg
#endif /* __INFO__ */
		      ) {
    char fn_[80];
    sprintf(fn_, "%s/%04d_%d.pgm", fn, field, component);
    FILE *fd  = fopen(fn_, "r");
    if(!fd) {
#if defined (__WARNING__)
      warning("%s: using \"/dev/zero\" instead of \"%s\"\n", msg, fn_);
#endif /* __INFO__ */
      fd = fopen("/dev/zero", "r");
    }
    motion::read(fd, data, blocks_in_y, blocks_in_x);
#if defined (__INFO__)
    info("%s: read %dx%d from \"%s\"\n", msg, blocks_in_y, blocks_in_x, fn_);
#endif /* __INFO__ */

    fclose(fd);
  }

  void write_component(TYPE **data,
		       int blocks_in_y,
		       int blocks_in_x,
		       char *fn,
		       int field,
		       int component
#if defined (__DEBUG__) || defined (__INFO__) || defined (__WARNING__)
		       ,
		       char *msg
#endif /* __INFO__ */
		       ) {
    char fn_[80];
    sprintf(fn_, "%s/%04d_%d.pgm", fn, field, component);
    FILE *fd  = fopen(fn_, "w");
#if defined (__DEBUG__)
    if(!fd) {
      error("%s: unable to create the file \"%s\" ... aborting!\n", msg, fn_);
      abort();
    }
#endif /* __DEBUG__ */
    motion::write(fd, data, blocks_in_y, blocks_in_x);
#if defined (__INFO__)
    info("%s: written %dx%d to \"%s\"\n", msg, blocks_in_y, blocks_in_x, fn_);
#endif /* __INFO__ */
    fclose(fd);
  }
#endif

};

