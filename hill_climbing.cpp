#include "hill_climbing.h"
#include "math.h"
#include <algorithm>

using namespace std;

double HillClimbing::function(double x)
{
	if (x<=1)
	{
		return 2 * x - x * x;
	}
	else
	{
		return 1;
	}
}

pair<vector<double>*, double>* HillClimbing::run(int k, int theta, double delta, vector<vector<int>*>* RR_sets){
	pair<vector<double>*, double>* res = new pair<vector<double>*, double>;
	res->first = new vector<double>;
	res->second = 0.0;

	bool RR_sets_null = (RR_sets == NULL);
	if(RR_sets_null){
		RR_sets = generate_RR_sets(theta);
	}

	vector<double>* s = new vector<double>;
	s->resize(RR_sets->size(), 1.0);
	int n = gf->getSize();

	if(type == 0){
		res->first->resize(n, 0.0);
		for(int i = 0; i < double(k) / delta; i++){
			iter_greedy(RR_sets, s, res->first, delta, n);
		}
	}
	else if(type == 1){
		int d = prob_strat_to_node->size();
		res->first->resize(d, 0.0);
		vector<vector<pair<int, int>*>*>* strat_to_node_and_RR = new vector<vector<pair<int, int>*>*>;
		for(int i = 0; i < d; i++){
			strat_to_node_and_RR->push_back(new vector<pair<int, int>*>);
		}
		for(int i = 0; i < RR_sets->size(); i++){
			vector<int>* RR_set = RR_sets->at(i);
			for(int j = 0; j < RR_set->size(); j++){
				int node = RR_set->at(j);
				for(int l = 0; l < node_to_strat->at(node)->size(); l++){
					strat_to_node_and_RR->at(node_to_strat->at(node)->at(l))->push_back(new pair<int, int>(i, node));
				}
			}
		}
		for(int i = 0; i < double(k) / delta; i++){
			iter_greedy_2(RR_sets, s, res->first, delta, n, strat_to_node_and_RR);
		}

		for(int i = 0; i < d; i++){
			vector<pair<int, int>*>* tmp_vec_pair = strat_to_node_and_RR->at(i);
			for(int j = 0; j < tmp_vec_pair->size(); j++){
				delete tmp_vec_pair->at(j);
			}
			delete strat_to_node_and_RR->at(i);
		}
		delete strat_to_node_and_RR;
	}

	for(int i = 0; i < s->size(); i++){
		res->second += 1 - s->at(i);
	}

	res->second *= double(n) / theta;

	if(RR_sets_null){
		for(int i = 0; i < RR_sets->size(); i++){
			delete RR_sets->at(i);
		}
		delete RR_sets;
	}
	delete s;
	return res;
}

void HillClimbing::iter_greedy(vector<vector<int>*>* RR_sets, vector<double>* s, vector<double>* value, double del, int n){
	vector<double>* delta = new vector<double>;

	delta->resize(n, 0.0);

	for(int i = 0; i < RR_sets->size(); i++){
		vector<int>* RR_set = RR_sets->at(i);
		for(int j = 0; j < RR_set->size(); j++){
			int node = RR_set->at(j);
			delta->at(node) += s->at(i) * (function(value->at(node) + del) - function(value->at(node))) / (1 - function(value->at(node)));
		}
	}

	int opt;
	double max_delta = -1;
	for(int i = 0; i < delta->size(); i++){
		if(delta->at(i) > max_delta){
			max_delta = delta->at(i);
			opt = i;
		}
	}

	double factor = (1 - function(value->at(opt) + del)) / (1 - function(value->at(opt)));
	for(int i = 0; i < RR_sets->size(); i++){
		vector<int>* RR_set = RR_sets->at(i);
		for(int j = 0; j < RR_set->size(); j++){
			int node = RR_set->at(j);
			if(node == opt){
				s->at(i) = s->at(i) * factor;
				break;
			}
		}
	}

	value->at(opt) += del;
	delete delta;
}

void HillClimbing::iter_greedy_2(vector<vector<int>*>* RR_sets, vector<double>* s, vector<double>* value, double del, int n, vector<vector<pair<int, int>*>*>* strat_to_node_and_RR){
	int d = strat_to_node_and_RR->size();
	vector<double>* delta = new vector<double>;
	delta->resize(d, 0.0);
	for(int i = 0; i < d; i++){
		int label = -1;
		double cur_delta = 1;
		vector<pair<int, int>*>* R_i = strat_to_node_and_RR->at(i);
		for(int j = 0; j < R_i->size(); j++){
			pair<int, int>* RR_and_node = R_i->at(j);
			int l = RR_and_node->first;
			int node = RR_and_node->second;
			if(l != label && label != -1){
				delta->at(i) += (1- cur_delta) * s->at(label);
				label = l;
				cur_delta = s->at(l);
			}
			else if(label == -1){
				label = l;
				cur_delta = s->at(l);
			}
			cur_delta *= pow(prob_strat_to_node->at(i)->at(node), value->at(i) + del) / pow(prob_strat_to_node->at(i)->at(node), value->at(i));
		}
		if(label != -1){
			delta->at(i) += (1- cur_delta) * s->at(label);
		}
	}

	int opt;
	double max_delta = -1;
	for(int i = 0; i < delta->size(); i++){
		if(delta->at(i) > max_delta){
			max_delta = delta->at(i);
			opt = i;
		}
	}

	for(int i = 0; i < RR_sets->size(); i++){
		for(int j = 0; j < RR_sets->at(i)->size(); j++){
			int node = RR_sets->at(i)->at(j);
			for(int l = 0; l < node_to_strat->at(node)->size(); l++){
				if(node_to_strat->at(node)->at(l) == opt){
					s->at(i) *= pow(prob_strat_to_node->at(opt)->at(node), value->at(opt) + del) / pow(prob_strat_to_node->at(opt)->at(node), value->at(opt));
				}
			}
		}
	}

	value->at(opt) += del;
	delete delta;
}

void HillClimbing::_generate_RR_set(int start_node, vector<bool>* active, vector<int>* RR_set){
	active->at(start_node) = true;
	RR_set->push_back(start_node);
	for(set<Edge>::iterator iter = gf->getIterFColBegin(start_node); iter != gf->getIterFColEnd(start_node); iter++){
		int target = iter->src;
		if(active->at(target)){
			continue;
		}
		double prob = iter->value;
		bool if_activated = ((rand() / (RAND_MAX+1.0)) < prob);
		if(if_activated){
			_generate_RR_set(target, active, RR_set);
		}
	}
}

vector<vector<int>*>* HillClimbing::generate_RR_sets(int theta){
	vector<vector<int>*>* res = new vector<vector<int>*>;

	int n = gf->getSize();

	for (int it = 0; it < theta; it++){
        vector<bool>* active = new vector<bool>;
        active->resize(n, false);
		vector<int>* RR_set = new vector<int>;
		int start_node = rand()%n;
		_generate_RR_set(start_node, active, RR_set);
		delete active;
		res->push_back(RR_set);
	}
	return res;
}

double CIMM::_lambda(double eps, int k, double delta, int n, double l, int d){
	return (2 + 2.0 / 3.0 * eps) * (double(k) / delta * log(d) + l * log(n) + log(log2(n))) * double(n) / (eps * eps);
}

double CIMM::_lambda2(double eps, int k, int n, double l, double delta, int d){
	double alpha = sqrt(l * log(n) + log(2));
	double beta = sqrt((1 - 1.0 / exp(1)) * (double(k) / delta * log(d) + l * log(n) + log2(n)));
	double alpha_beta = ((1 - 1.0 / exp(1))) * alpha + beta;
	return 2 * n * (alpha_beta * alpha_beta) / (eps * eps / 2.0);
}

vector<double>* CIMM::run(int k, double delta, double eps, double l){
	vector<vector<int>*>* RR_sets = new vector<vector<int>*>;
	int n = gf->getSize();
	double LB = 1.0;
	eps = eps * sqrt(2);
	//l = l + log(2) / log(n);
	int d;
	if (this->hill_climbing->type == 0) {
		d = n;
	}
	else if (this->hill_climbing->type == 1) {
		d = this->hill_climbing->prob_strat_to_node->size();
	}
	else {
		d = -1;
	}

	double gamma, lgamma, rgamma;

	lgamma = 0;
	rgamma = 2;

	// doubling search to find the correct right-side gamma;
	while (ceil(_lambda2(eps, k, n, l + rgamma, delta, d)) > pow(n, rgamma)) {
		lgamma = rgamma;
		rgamma = rgamma * 2;
	}

	// binary search to find the gamma in the middle;
	while (rgamma > lgamma + 0.1) {
		gamma = (lgamma + rgamma) / 2;
		if (ceil(_lambda2(eps, k, n, l + gamma, delta, d)) > pow(n, gamma)) {
			lgamma = gamma;
		}
		else {
			rgamma = gamma;
		}
	}

	l = l + rgamma;
	l = l + log(2) / log(n);

	int max_round = fmax(fmax(log2((double)n), 1.0) - 1.0, 1.0);

	for(int i = 1; i <  max_round + 1; i++){
		cout << "round " << i << ": ";
		double y = double(n) / pow(2, i);
		double lambda = _lambda(eps, k, delta, n, l, d);
		double theta = lambda / y;
		cout << "theta=" << theta << endl;
		int j = RR_sets->size();
		while(j < theta){
			generate_RR_sets(1, RR_sets);
			j = RR_sets->size();
		}
		pair<vector<double>*, double>* res = hill_climbing->run(k, RR_sets->size(), delta, RR_sets);
		if(res->second >= (1 + eps) * y){
			LB = res->second / (1 + eps);
			break;
		}
	}

	cout << "Final Step: ";

	//int theta = int(_lambda2(eps, k, n, l, delta, d) / LB);
	//vector<vector<int>*>* RR_sets_2 = new vector<vector<int>*>;//Regenerate a RR set.
	//int j = 0;
	//while (j < theta) {
	//	RR_sets_2 = generate_RR_sets(1, RR_sets_2);
	//	j = RR_sets_2->size();
	//}

	//cout << "RR_sets' size=" << RR_sets_2->size() << endl;

	//pair<vector<double>*, double>* res = hill_climbing->run(k, RR_sets_2->size(), delta, RR_sets_2);
	//for (int i = 0; i < RR_sets->size(); i++) {
	//	delete RR_sets->at(i);
	//}
	//for (int i = 0; i < RR_sets_2->size(); i++) {
	//	delete RR_sets_2->at(i);
	//}
	//delete RR_sets;
	//delete RR_sets_2;

	//return res->first;

	int theta = int(_lambda2(eps, k, n, l, delta, d) / LB);
	//vector<vector<int>*>* RR_sets_2 = new vector<vector<int>*>;//Regenerate a RR set.
	int j = RR_sets->size();
	while(j < theta){
		RR_sets = generate_RR_sets(1, RR_sets);
		j = RR_sets->size();
	}

	cout << "RR_sets' size=" << RR_sets->size() << endl;

	pair<vector<double>*, double>* res = hill_climbing->run(k, RR_sets->size(), delta, RR_sets);
	for(int i = 0; i < RR_sets->size(); i++){
		delete RR_sets->at(i);
	}
	//for(int i = 0; i < RR_sets_2->size(); i++){
	//	delete RR_sets_2->at(i);
	//}
	delete RR_sets;
	//delete RR_sets_2;

	return res->first;
}

void CIMM::_generate_RR_set(int start_node, vector<bool>* active, vector<int>* RR_set){
	active->at(start_node) = true;
	RR_set->push_back(start_node);
	for(set<Edge>::iterator iter = gf->getIterFColBegin(start_node); iter != gf->getIterFColEnd(start_node); iter++){
		int target = iter->src;
		if(active->at(target)){
			continue;
		}
		double prob = iter->value;
		bool if_activated = ((rand() / (RAND_MAX+1.0)) < prob);
		if(if_activated){
			_generate_RR_set(target, active, RR_set);
		}
	}
}

vector<vector<int>*>* CIMM::generate_RR_sets(int theta, vector<vector<int>*>* RR_sets){
	if(RR_sets == NULL){
		RR_sets = new vector<vector<int>*>;
	}

	int n = gf->getSize();

	for (int it = 0; it < theta; it++){
        vector<bool>* active = new vector<bool>;
        active->resize(n, false);
		vector<int>* RR_set = new vector<int>;
		int start_node = rand()%n;
		_generate_RR_set(start_node, active, RR_set);
		delete active;
		RR_sets->push_back(RR_set);
	}
	return RR_sets;
}






double CIMM_pseudo::_lambda(double eps, int k, double delta, int n, double l, int d) {
	return (2 + 2.0 / 3.0 * eps) * (int(k/delta) * log(d) + l * log(n) + log(log2(n))) * double(n) / (eps * eps);
}

double CIMM_pseudo::_lambda2(double eps, int k, int n, double l, double delta, int d) {
	double alpha = sqrt(l * log(n) + log(2));
	double beta = sqrt((1 - 1.0 / exp(1)) * (int(k/delta) * log(d) + l * log(n) + log2(n)));
	double alpha_beta = ((1 - 1.0 / exp(1))) * alpha + beta;
	return 2 * n * (alpha_beta * alpha_beta) / (eps * eps/2);
}

//revise needed
//classical greedy alg for picking pseudo nodes (linear complexity)
//return strategy and Fr
pair<vector<double>*, double>* CIMM_pseudo::node_selection_pseudo(vector<vector<int>*>* RR_sets, int k, double delta) {
	pair<vector<double>*, double>* res = new pair<vector<double>*, double>;
	res->first = new vector<double>;
	res->second = 0.0;

	//bool RR_sets_null = (RR_sets == NULL);
	//if (RR_sets_null) {
	//	RR_sets = generate_RR_sets(theta);
	//}

	//vector<double>* s = new vector<double>;
	//s->resize(RR_sets->size(), 1.0);
	//int n = gf->getSize();

	int n = gf->getSize();
	int nR = RR_sets->size();
	int nP = int(n / delta);
	res->first->resize(n, 0.0);
	
	vector<vector<int>*>* node_to_RR = new vector<vector<int>*>(nP);
	for (int i = 0; i < nP; i++)
	{
		node_to_RR->at(i) = new vector<int>;
	}
	for (int i = 0; i < nR; i++)
	{
		for (int j = 0; j < RR_sets->at(i)->size(); j++)
		{
			int node = RR_sets->at(i)->at(j);
			node_to_RR->at(node)->push_back(i);
		}
	}

	for (int iter = 0; iter < int(k/delta); iter++)
	{
		int max = node_to_RR->at(0)->size();
		int imax = 0;
		for (int i = 1; i < nP; i++)
		{
			int siz = node_to_RR->at(i)->size();
			if (max < siz)
			{
				max = siz;
				imax = i;
			}
		}
		res->second += max;
		res->first->at(imax / int(1 / delta)) += delta;

		for (int i = 0; i < max; i++)
		{
			int rr = node_to_RR->at(imax)->at(i);
			for (int j = 0; j < RR_sets->at(rr)->size(); j++)
			{
				int node = RR_sets->at(rr)->at(j);
				if (node != imax)
				{
					node_to_RR->at(node)->erase(remove(node_to_RR->at(node)->begin(), node_to_RR->at(node)->end(), rr), node_to_RR->at(node)->end());
				}
			}
		}
		node_to_RR->at(imax)->clear();
	}


	for (int i = 0; i < node_to_RR->size(); i++)
	{
		delete node_to_RR->at(i);
	}
	delete node_to_RR;


	//res->first->resize(n, 0.0);
	//res->second = 0.0;
	////build table of size #RR_sets * #pseudo_nodes, tabl[i][j]=1 iff pseudo_nodes i is contained in jth RR set
	//int RR_sets_size = RR_sets->size();
	//int pseudo_node_size = int(1 / delta)*n;
	//int **tabl;
	//tabl = new int* [RR_sets_size];
	//for (int i = 0; i < RR_sets_size; i++)
	//{
	//	tabl[i] = new int[pseudo_node_size];
	//	for (int j = 0; j < pseudo_node_size; j++)
	//	{
	//		tabl[i][j] = 0;
	//	}
	//}
	//for (int i = 0; i < RR_sets->size(); i++) {
	//	vector<int>* RR_set = RR_sets->at(i);
	//	for (int j = 0; j < RR_set->size(); j++) {
	//		int node = RR_set->at(j);
	//		tabl[i][node] = 1;
	//	}
	//}

	//vector<int>* counter = new vector<int>(pseudo_node_size);
	//for (int iter = 0; iter < int(k/delta); iter++)
	//{
	//	//calculate #RR sets covered for each pseudo-node
	//	for (int i = 0; i < pseudo_node_size; i++)
	//	{
	//		counter->at(i) = 0;
	//	}
	//	for (int i = 0; i < counter->size(); i++)
	//	{
	//		for (int j = 0; j < RR_sets_size; j++)
	//		{
	//			if (tabl[j][i] == 1)
	//			{
	//				counter->at(i) += 1;
	//			}
	//		}
	//	}
	//	//pick the one with largest covering
	//	auto maxpos = max_element(counter->begin(), counter->end());
	//	int cover = *maxpos;
	//	res->second += cover;
	//	int lnode = distance(counter->begin(), maxpos);
	//	res->first->at(lnode / int(1 / delta)) += delta;//k->1
	//	//make col of lnode 0 and all RR sets lnode covers 0
	//	for (int i = 0; i < RR_sets_size; i++)
	//	{
	//		if (tabl[i][lnode] == 1)
	//		{
	//			//tabl[i][lnode] = 0;
	//			for (int j = 0; j < pseudo_node_size; j++)
	//			{
	//				tabl[i][j] = 0;
	//			}
	//		}
	//	}
	//}
	//delete counter;
	//delete tabl;

	res->second = res->second / nR * n;//* (1 + int(1/delta));
	return res;
}

vector<double>* CIMM_pseudo::run(int k, double delta, double eps, double l) {
	vector<vector<int>*>* RR_sets = new vector<vector<int>*>;
	int n = gf->getSize();
	int n_pseudo = n;// * (1 + int(1 / delta));
	double LB = 1.0;
	eps = eps * sqrt(2);
	//l = l * (1 + log(2) / log(n_pseudo));
	int max_round = fmax(fmax(log2((double)n_pseudo), 1.0) - 1.0, 1.0);

	int d;

	if (this->hill_climbing->type == 0) {
		d = n_pseudo;
		double gamma, lgamma, rgamma;

		lgamma = 0;
		rgamma = 2;

		// doubling search to find the correct right-side gamma;
		while (ceil(_lambda2(eps, k, n, l + rgamma, delta, d)) > pow(n, rgamma)) {
			lgamma = rgamma;
			rgamma = rgamma * 2;
		}

		// binary search to find the gamma in the middle;
		while (rgamma > lgamma + 0.1) {
			gamma = (lgamma + rgamma) / 2;
			if (ceil(_lambda2(eps, k, n, l + gamma, delta, d)) > pow(n, gamma)) {
				lgamma = gamma;
			}
			else {
				rgamma = gamma;
			}
		}

		l = l + rgamma;
		l = l + log(2) / log(n);

		int lengt = int(1 / delta);
		vector<double> pseudo_value(lengt);
		for (int i = 0; i < lengt; i++)
		{
			pseudo_value[i] = hill_climbing->function((i + 1)*delta);
			//pseudo_value[i] = hill_climbing->function((i + 1)*delta) - hill_climbing->function(i*delta);
		}
		for (int i = 1; i < max_round + 1; i++) {
			cout << "round " << i << ": ";
			double y = double(n_pseudo) / pow(2, i);
			double lambda = _lambda(eps, k, delta, n_pseudo, l, d);
			double theta = lambda / y;
			cout << "theta=" << theta << endl;
			int j = RR_sets->size();
			while (j < theta) {
				generate_RR_sets(pseudo_value, 1, RR_sets);
				j = RR_sets->size();
			}
			pair<vector<double>*, double>* res = node_selection_pseudo(RR_sets, k, delta);
			if (res->second >= (1 + eps) * y) {//revised
				LB = res->second / (1 + eps);
				break;
			}
		}

		cout << "Final Step: ";

		int theta = int(_lambda2(eps, k, n_pseudo, l, delta, d) / LB);
		//vector<vector<int>*>* RR_sets_2 = new vector<vector<int>*>;//Regenerate a RR set.
		int j = RR_sets->size();
		while (j <= theta) {
			generate_RR_sets(pseudo_value, 1, RR_sets);
			j = RR_sets->size();
		}

		cout << "RR_sets' size=" << RR_sets->size() << endl;

		pair<vector<double>*, double>* res = node_selection_pseudo(RR_sets, k, delta);
		for (int i = 0; i < RR_sets->size(); i++) {
			delete RR_sets->at(i);
		}
		//for (int i = 0; i < RR_sets_2->size(); i++) {
		//	delete RR_sets_2->at(i);
		//}
		delete RR_sets;
		//delete RR_sets_2;

		return res->first;
	}
	else if (this->hill_climbing->type == 1) {
		d = this->hill_climbing->prob_strat_to_node->size();

		double gamma, lgamma, rgamma;

		lgamma = 0;
		rgamma = 2;

		// doubling search to find the correct right-side gamma;
		while (ceil(_lambda2(eps, k, n, l + rgamma, delta, d)) > pow(n, rgamma)) {
			lgamma = rgamma;
			rgamma = rgamma * 2;
		}

		// binary search to find the gamma in the middle;
		while (rgamma > lgamma + 0.1) {
			gamma = (lgamma + rgamma) / 2;
			if (ceil(_lambda2(eps, k, n, l + gamma, delta, d)) > pow(n, gamma)) {
				lgamma = gamma;
			}
			else {
				rgamma = gamma;
			}
		}

		l = l + rgamma;
		l = l + log(2) / log(n);

		for (int i = 1; i < max_round + 1; i++) {
			cout << "round " << i << ": ";
			double y = double(n_pseudo) / pow(2, i);
			double lambda = _lambda(eps, k, delta, n_pseudo, l, d);
			double theta = lambda / y;
			cout << "theta=" << theta << endl;
			int j = RR_sets->size();
			while (j < theta) {
				generate_RR_sets_2(1, RR_sets, k, delta);
				j = RR_sets->size();
			}
			pair<vector<double>*, double>* res = node_selection_pseudo_2(RR_sets, d, k, delta);
			if (res->second >= (1 + eps) * y) {//revised
				LB = res->second / (1 + eps);
				break;
			}
		}

		cout << "Final Step: ";

		int theta = int(_lambda2(eps, k, n_pseudo, l, delta, d) / LB);
		//vector<vector<int>*>* RR_sets_2 = new vector<vector<int>*>;//Regenerate a RR set.
		int j = RR_sets->size();
		while (j <= theta) {
			generate_RR_sets_2(1, RR_sets, k ,delta);
			j = RR_sets->size();
		}

		cout << "RR_sets' size=" << RR_sets->size() << endl;

		pair<vector<double>*, double>* res = node_selection_pseudo_2(RR_sets, d, k, delta);
		for (int i = 0; i < RR_sets->size(); i++) {
			delete RR_sets->at(i);
		}
		//for (int i = 0; i < RR_sets_2->size(); i++) {
		//	delete RR_sets_2->at(i);
		//}
		delete RR_sets;
		//delete RR_sets_2;

		return res->first;
	}
	else {
		d = -1;
	}
}

void CIMM_pseudo::_generate_RR_set(vector<double> pseudo_value, int start_node, vector<bool>* active, vector<int>* RR_set) {
	active->at(start_node) = true;
	//RR_set->push_back(start_node);

	//add pesudo nodes into RR set
	int index;
	int lengt = pseudo_value.size();
	
	double t = double(rand()) / (RAND_MAX+1.0);

	if (t <= pseudo_value[0])
	{
		index = 0;
	}
	else
	{
		int startindex = 0;
		int endindex = lengt - 1;
		index = (startindex + endindex) / 2;
		while (startindex < endindex - 1)
		{
			if (t <= pseudo_value[index])
			{
				endindex = index;
			}
			else
			{
				startindex = index;
			}
			index = (startindex + endindex) / 2;
		}
		index = index + 1;
	}

	//bisection 
	
	//for (int i = 0; i < lengt; i++)
	//{
	//	if (t <= pseudo_value[i]) {
	//		index = i;
	//		break;
	//	}
	//	else
	//	{
	//		t -= pseudo_value[i];
	//	}
	//}

	//if (index != index_i)
	//{
	//	cout << "error" << endl;
	//}

	RR_set->push_back(start_node*lengt + index);

	for (set<Edge>::iterator iter = gf->getIterFColBegin(start_node); iter != gf->getIterFColEnd(start_node); iter++) {
		int target = iter->src;
		if (active->at(target)) {
			continue;
		}
		double prob = iter->value;
		bool if_activated = ((rand() / (RAND_MAX + 1.0)) < prob);
		if (if_activated) {
			_generate_RR_set(pseudo_value, target, active, RR_set);
		}
	}
}

vector<vector<int>*>* CIMM_pseudo::generate_RR_sets(vector<double> pseudo_value, int theta, vector<vector<int>*>* RR_sets) {
	
	if (RR_sets == NULL) {
		RR_sets = new vector<vector<int>*>;
	}

	int n = gf->getSize();

	for (int it = 0; it < theta; it++) {
		vector<bool>* active = new vector<bool>;
		active->resize(n, false);
		vector<int>* RR_set = new vector<int>;
		int start_node = rand() % n;
		_generate_RR_set(pseudo_value, start_node, active, RR_set);
		delete active;
		RR_sets->push_back(RR_set);
	}
	return RR_sets;
}

void CIMM_pseudo::_generate_RR_set_2(int start_node, vector<bool>* active, vector<int>* RR_set, int k, double delta) {
	active->at(start_node) = true;
	if (hill_climbing->node_to_strat->at(start_node)->size() > 0)
	{
		int start_strategy = hill_climbing->node_to_strat->at(start_node)->at(0);
		double start_prob = hill_climbing->prob_strat_to_node->at(start_strategy)->at(start_node);
		int lengt = int(k / delta);
		vector<double>* pseudo_value = new vector<double>(lengt);
		//vector<double>* pseudo_value_b = new vector<double>(lengt);
		for (int i = 0; i < lengt; i++)
		{
			pseudo_value->at(i) = 1 - pow(start_prob, (i+1) * delta);
			//pseudo_value_b->at(i) = pow(start_prob, i * delta) - pow(start_prob, (i + 1) * delta);
		}
		//RR_set->push_back(start_node);

		//add pesudo nodes into RR set
		int index = -1;
		//int index_i = -1;
		//int lengt = int(k/delta);

		double t = double(rand()) / (RAND_MAX + 1.0);
		//double tt = t;

		//for (int i = 0; i < lengt; i++)
		//{
		//	if (tt <= pseudo_value_b->at(i)) {
		//		index_i = i;
		//		break;
		//	}
		//	else
		//	{
		//		//index++;
		//		tt -= pseudo_value_b->at(i);
		//	}
		//}

		if (t <= pseudo_value->at(0))
		{
			index = 0;
		}
		else if (t <= pseudo_value->at(lengt - 1))
		{
			int startindex = 0;
			int endindex = lengt - 1;
			index = (startindex + endindex) / 2;
			while (startindex < endindex - 1)
			{
				if (t <= pseudo_value->at(index))
				{
					endindex = index;
				}
				else
				{
					startindex = index;
				}
				index = (startindex + endindex) / 2;
			}
			index = index + 1;
		}

		//if (index != index_i)
		//{
		//	cout << "error" << endl;
		//}

		if (index != -1)
		{
			RR_set->push_back(start_strategy*lengt + index);
		}
	}


	for (set<Edge>::iterator iter = gf->getIterFColBegin(start_node); iter != gf->getIterFColEnd(start_node); iter++) {
		int target = iter->src;
		if (active->at(target)) {
			continue;
		}
		double prob = iter->value;
		bool if_activated = ((rand() / (RAND_MAX + 1.0)) < prob);
		if (if_activated) {
			_generate_RR_set_2(target, active, RR_set, k, delta);
		}
	}
}

vector<vector<int>*>* CIMM_pseudo::generate_RR_sets_2(int theta, vector<vector<int>*>* RR_sets, int k, double delta) {

	if (RR_sets == NULL) {
		RR_sets = new vector<vector<int>*>;
	}

	int n = gf->getSize();

	for (int it = 0; it < theta; it++) {
		vector<bool>* active = new vector<bool>;
		active->resize(n, false);
		vector<int>* RR_set = new vector<int>;
		int start_node = rand() % n;
		_generate_RR_set_2(start_node, active, RR_set, k, delta);
		delete active;
		RR_sets->push_back(RR_set);
	}
	return RR_sets;
}

pair<vector<double>*, double>* CIMM_pseudo::node_selection_pseudo_2(vector<vector<int>*>* RR_sets, int d, int k, double delta) {
	pair<vector<double>*, double>* res = new pair<vector<double>*, double>;
	res->first = new vector<double>;
	res->second = 0.0;

	//bool RR_sets_null = (RR_sets == NULL);
	//if (RR_sets_null) {
	//	RR_sets = generate_RR_sets(theta);
	//}

	//vector<double>* s = new vector<double>;
	//s->resize(RR_sets->size(), 1.0);
	//int n = gf->getSize();

	int n = gf->getSize();
	int nR = RR_sets->size();
	int nP = int(d * k / delta);
	res->first->resize(d, 0.0);

	vector<vector<int>*>* node_to_RR = new vector<vector<int>*>(nP);
	for (int i = 0; i < nP; i++)
	{
		node_to_RR->at(i) = new vector<int>;
	}
	for (int i = 0; i < nR; i++)
	{
		for (int j = 0; j < RR_sets->at(i)->size(); j++)
		{
			int node = RR_sets->at(i)->at(j);
			node_to_RR->at(node)->push_back(i);
		}
	}

	for (int iter = 0; iter < int(k / delta); iter++)
	{
		int max = node_to_RR->at(0)->size();
		int imax = 0;
		for (int i = 1; i < nP; i++)
		{
			int siz = node_to_RR->at(i)->size();
			if (max < siz)
			{
				max = siz;
				imax = i;
			}
		}
		res->second += max;
		res->first->at(imax / int(k / delta)) += delta;

		for (int i = 0; i < max; i++)
		{
			int rr = node_to_RR->at(imax)->at(i);
			for (int j = 0; j < RR_sets->at(rr)->size(); j++)
			{
				int node = RR_sets->at(rr)->at(j);
				if (node != imax)
				{
					node_to_RR->at(node)->erase(remove(node_to_RR->at(node)->begin(), node_to_RR->at(node)->end(), rr), node_to_RR->at(node)->end());
				}
			}
		}
		node_to_RR->at(imax)->clear();
	}


	for (int i = 0; i < node_to_RR->size(); i++)
	{
		delete node_to_RR->at(i);
	}
	delete node_to_RR;
	res->second = res->second / nR * n;//* (1 + int(1/delta));
	return res;
}