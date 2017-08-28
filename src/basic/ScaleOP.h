#ifndef SCALEOP
#define SCALEOP


#include "Eigen/Dense"
#include "MyLib.h"
#include "Node.h"
#include "Graph.h"

class PScaleNode : public Node {
public:
public:
	PNode in1, in2;
public:
	PScaleNode() : Node() {
		in1 = NULL;
		in2 = NULL;
		node_type = "scale";
	}
public:
	virtual inline void clearValue() {
		Node::clearValue();
		in1 = NULL;
		in2 = NULL;
	}

public:
	void forward(Graph *cg, PNode x1, PNode x2) {
		in1 = x1;
		in2 = x2;
		degree = 0;
		if (in1->dim != 1) {
			cout << "scale node, dim of x1 must be 1." << endl;
		}
		x1->addParent(this);
		x2->addParent(this);
		cg->addNode(this);
	}

public:
	inline void compute() {
		val.tmat() = in1->val.mat() * in2->val.tmat();
	}

	void backward() {
		in1->loss.mat() += loss.tmat() * in2->val.mat();
		in2->loss.mat() += loss.mat() * in1->val.mat();
	}

public:
	// better to rewrite for deep understanding
	inline bool typeEqual(PNode other) {
		return Node::typeEqual(other);
	}

	inline PExecute generate(bool bTrain);
};

class PScaleExecute :public Execute {
public:
	bool bTrain;
public:
	inline void  forward() {
		int count = batch.size();
		//#pragma omp parallel for schedule(static,1)
		for (int idx = 0; idx < count; idx++) {
			PScaleNode* ptr = (PScaleNode*)batch[idx];
			ptr->compute();
			ptr->forward_drop(bTrain);
		}
	}

	inline void backward() {
		int count = batch.size();
		//#pragma omp parallel for schedule(static,1)
		for (int idx = 0; idx < count; idx++) {
			PScaleNode* ptr = (PScaleNode*)batch[idx];
			ptr->backward_drop();
			ptr->backward();
		}
	}
};

inline PExecute PScaleNode::generate(bool bTrain) {
	PScaleExecute* exec = new PScaleExecute();
	exec->batch.push_back(this);
	exec->bTrain = bTrain;
	return exec;
}

#endif // SCALE
