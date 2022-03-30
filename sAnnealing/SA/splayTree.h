#ifndef _SPLAY_TREE_H
#define _SPLAY_TREE_H

typedef struct sTree *sTree;
typedef struct node *node;

sTree
allocSTree(int n
           );

sTree
dupSTree(sTree t
        );

sTree
cpySTree(sTree dst,
         sTree src
        );

void
freeSTree(sTree t
          );

/* Turns into an empty tree */
void
clearTree(sTree t
          );

int
size(sTree t
     );

enum childI {left, right};

/* Return the key associated with this node */
int
key(sTree t,
    node n
    );

/* Return the value stored in this node. */
int
value(sTree t,
      node n,
      int *cv
      );

/* Return node pointer by index */
node
getNode(sTree t,
        int i
        );

/* Return node pointer in order */
node
getNodeInOrder(sTree t,
	       int i
	       );

/* Get the inoder keys */
int *
getInorder(sTree t,
           int *L,
	   int *map,
	   enum childI d
           );

/* Insert by key */
void
insertKey(sTree t,
	  int k
	  );

/* Insert inorder */
void
insertInorderKey(sTree t,
		 int p, /* position */
		 int k
		 );

/* Insert by relative position */
void
insertN(sTree t,
        node n, /* Reference node */
        int i, /* Node index on t */
        enum childI d /* Direction of reference */
        );

/* Make the node corresponding to i the new root */
void
reRoot(sTree t,
       int i, /* New root index */
       enum childI d /* Direction of reference */
       );

void
removeN(sTree t,
        node n /* Node to remove */
        );

void
splitSt(sTree t,
	int k, /* The key */
	enum childI d /* Direction to keep */
	);

/* Next or prev element, depending on direction. */
node
diressor(sTree t,
	 node n, /* Reference node */
	 enum childI d
	 );

/* Maximum or minimum, depending on direction. */
node
dirum(sTree t,
      enum childI d
      );

/* Get the floor on key space */
void
roundSt(sTree t,
        int k, /* The key */
	node *floor,
	node *ceil
        );

#endif /* _SPLAY_TREE_H */
