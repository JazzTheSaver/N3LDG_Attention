#ifndef ATTENTIONBUILDER
#define ATTENTIONBUILDER

#include "SoftmaxOP.h"
#include "PMultiOP.h"
#include "ScaleOP.h"

struct AttentionVParams{
	BiParams _att_weight;

	AttentionVParams(){
	}

	inline void exportAdaParams(ModelUpdate& ada) {
		_att_weight.exportAdaParams(ada);
	}

	inline void initial(int nHSize, int nGSize) {
		_att_weight.initial(nHSize, nHSize, nGSize, false);
	}

	inline void save(std::ofstream &os) const {
		_att_weight.save(os);
	}

	inline void load(std::ifstream &is) {
		_att_weight.load(is);
	}

	inline int outDim() {
		return _att_weight.W2.outDim();
	}
};

class AttentionVBuilder {
public:
	AttentionVParams* _params;
	vector<BiNode> _bilayer;
	SoftmaxBuilder _softmaxbuilder;
	vector<PMultiNode> _scales;
	PAddNode _output;
	int _dim;
	
	AttentionVBuilder(){
		clear();
	}

	~AttentionVBuilder() {
		clear();
	}

public:
	inline void resize(int maxsize) {
		_bilayer.resize(maxsize);
		_softmaxbuilder.resize(maxsize);
		_scales.resize(maxsize);
	}

	inline void clear() {
		_bilayer.clear();
		_softmaxbuilder.clear();
		_scales.clear();
	}

	inline void init(AttentionVParams* param, dtype dropout) {
		this->_params = param;
		_dim = _params->outDim();
		_softmaxbuilder.init(_dim);
		int maxsize = _scales.size();
		for(int idx = 0; idx < maxsize; idx++) { 
			_bilayer[idx].setParam(&_params->_att_weight);
			_bilayer[idx].init(_dim, dropout);
			_scales[idx].init(_dim, -1);
		}
		_output.init(_dim, -1);
	}

	inline void forward(Graph *cg, const vector<PNode>& x, PNode guide) { 
		if (x.size() == 0) {
			std::cout << "empty inputs for attention builder operation" << std::endl;
			return;
		}

		int nSize = x.size();
		if (x[0]->val.dim != _dim) {
			std::cout << "input dim dose not match for attention builder operation" << std::endl;
			return;
		}

		for (int idx = 0; idx < nSize; idx++) {
			_bilayer[idx].forward(cg, x[idx], guide);
		}

		_softmaxbuilder.forward(cg, getPNodes(_bilayer, nSize));
		for(int idx = 0; idx < nSize; idx++) {
			_scales[idx].forward(cg, x[idx], &_softmaxbuilder._output[idx]);
		}
		_output.forward(cg, getPNodes(_scales, nSize));
	}
};


struct AttentionParams{
	BiParams _att_weight;

	AttentionParams(){
	}

	inline void exportAdaParams(ModelUpdate& ada) {
		_att_weight.exportAdaParams(ada);
	}

	inline void initial(int nHSize, int nGSize) {
		_att_weight.initial(1, nHSize, nGSize, false);
	}

	inline void save(std::ofstream &os) const {
		_att_weight.save(os);
	}

	inline void load(std::ifstream &is) {
		_att_weight.load(is);
	}

	inline int outDim() {
		return _att_weight.W2.outDim();
	}

	inline int in1Dim() {
		return _att_weight.W1.inDim();
	}
};

class AttentionBuilder {
public:
	AttentionParams* _params;
	vector<BiNode> _bilayer;
	SoftmaxBuilder _softmaxbuilder;
	vector<PScaleNode> _scales;
	PAddNode _output;
	int _dim;

	AttentionBuilder() {
		clear();
	}

	~AttentionBuilder() {
		clear();
	}

public:
	inline void resize(int maxsize) {
		_bilayer.resize(maxsize);
		_softmaxbuilder.resize(maxsize);
		_scales.resize(maxsize);
	}

	inline void clear() {
		_bilayer.clear();
		_softmaxbuilder.clear();
		_scales.clear();
	}

	inline void init(AttentionParams* param, dtype dropout) {
		this->_params = param;
		_dim = _params->in1Dim();
		_softmaxbuilder.init(1);
		int maxsize = _scales.size();
		for (int idx = 0; idx < maxsize; idx++) {
			_bilayer[idx].setParam(&_params->_att_weight);
			_bilayer[idx].init(1, dropout);
			_scales[idx].init(_dim, -1);
		}
		_output.init(_dim, -1);
	}

	inline void forward(Graph *cg, const vector<PNode>& x, PNode guide) {
		if (x.size() == 0) {
			std::cout << "empty inputs for attention builder operation" << std::endl;
			return;
		}

		int nSize = x.size();
		if (x[0]->val.dim != _dim) {
			std::cout << "input dim dose not match for attention builder operation" << std::endl;
			return;
		}

		for (int idx = 0; idx < nSize; idx++) {
			_bilayer[idx].forward(cg, x[idx], guide);
		}

		_softmaxbuilder.forward(cg, getPNodes(_bilayer, nSize));
		for (int idx = 0; idx < nSize; idx++) {
			_scales[idx].forward(cg,  &_softmaxbuilder._output[idx], x[idx]);
		}
		_output.forward(cg, getPNodes(_scales, nSize));
	}

};

#endif // !ATTENTIONBUILDER
