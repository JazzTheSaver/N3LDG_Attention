#ifndef SELECTIION
#define SELECTIION

class SelectionNode :public Node{
public:
	PNode in;
	int start_pos;

public:
	SelectionNode() : Node() {
		in = NULL;
		start_pos = 0;
		node_type = "selection";
	}

public:
	virtual inline void clearValue() {
		Node::clearValue();
		in = NULL;
		start_pos = 0;
	}

public:
	inline PExecute generate(bool bTrain);

	// better to rewrite for deep understanding
	inline bool typeEqual(PNode other) {
		return Node::typeEqual(other);
	}


public:
	void forward(Graph *cg, PNode x, const int& start) {
		in = x;
		start_pos = start;
		if (start_pos < 0 || start_pos + dim > in->dim) {
			std::cout << "error: position overflow!" << std::endl;
			return;
		}
		degree = 0;
		in->addParent(this);
		cg->addNode(this);
	}
	
	inline void compute(){
		int end_pos = start_pos + dim;
		int offset = 0;
		for (int idx = start_pos; idx < end_pos; idx++) {
			val[offset] = in->val[idx];
			offset++;
		}
	}

	void backward() {
		int end_pos = start_pos + dim;
		int offset = 0;
		for (int idx = start_pos; idx < end_pos; idx++) {
			in->loss[idx] += loss[offset];
			offset++;
		}
	}
};

class SelectionExecute : public Execute {
public:
	bool bTrain;
public:
	inline void  forward() {
		int count = batch.size();
		//#pragma omp parallel for schedule(static,1)
		for (int idx = 0; idx < count; idx++) {
			SelectionNode* ptr = (SelectionNode*)batch[idx];
			ptr->compute();
			ptr->forward_drop(bTrain);
		}
	}

	inline void backward() {
		int count = batch.size();
		//#pragma omp parallel for schedule(static,1)
		for (int idx = 0; idx < count; idx++) {
			SelectionNode* ptr = (SelectionNode*)batch[idx];
			ptr->backward_drop();
			ptr->backward();
		}
	}
};

inline PExecute SelectionNode::generate(bool bTrain) {
	SelectionExecute* exec = new SelectionExecute();
	exec->batch.push_back(this);
	exec->bTrain = bTrain;
	return exec;
}

#endif
