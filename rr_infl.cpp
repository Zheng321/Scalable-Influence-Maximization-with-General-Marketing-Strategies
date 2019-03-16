
#include <iostream>
#include <vector>
#include <set>
#include <random>
#include <functional>
#include <algorithm>
#include <cassert>

#include "rr_infl.h"
#include "reverse_general_cascade.h"
#include "graph.h"
#include "event_timer.h"
#include "common.h"

using namespace std;




// define a comparator for counts
struct CountComparator
{
public:
	vector<int>& counts;
	CountComparator(vector<int>& c) :counts(c){}
public:
	bool operator () (int a, int b) {
		return (counts[a] <= counts[b]);
	}
};

struct dCountComparator
{
public:
	vector<double>& counts;
	dCountComparator(vector<double>& c) :counts(c){}
public:
	bool operator () (double a, double b) {
		return (counts[a] <= counts[b]);
	}
};



void RRInflBase::InitializeConcurrent()
{
	if (isConcurrent) 
	{
#ifdef MI_USE_OMP
		/////////////////////////////////////////////
		// run concurrently
		const double DYNAMIC_RATIO = 0.25;
		int maxThreads = omp_get_max_threads();
		omp_set_num_threads(maxThreads);
		int dynamicThreads = (int)(maxThreads * DYNAMIC_RATIO);
		omp_set_dynamic(dynamicThreads);
		
		cout << "== Turn on omp optimization: == " << endl;
		cout << "#Max Threads = " << omp_get_max_threads() << "\t#Dynamic Threads = " << omp_get_dynamic() << endl;
#else
        cout << "== omp is not supported or enabled == " << endl;
#endif
    }
}


void RRInflBase::_AddRRSimulation(size_t num_iter, 
							cascade_type& cascade, 
							std::vector< RRVec >& refTable,
							std::vector<int>& refTargets)
{
	vector<int> edgeVisited; // discard
	_AddRRSimulation(num_iter, cascade, refTable, refTargets, edgeVisited);
}

void RRInflBase::_AddRRSimulation(size_t num_iter, 
							  cascade_type& cascade, 
							  std::vector< RRVec >& refTable,
							  std::vector<int>& refTargets,
							  std::vector<int>& refEdgeVisited)
{
#ifdef MI_USE_OMP
	if (!isConcurrent) {
#endif
		// run single thread

		for (size_t iter = 0; iter < num_iter; ++iter) {
			int id = cascade.GenRandomNode();
			int edgeVisited;
			cascade.ReversePropagate(1, id, refTable, edgeVisited);

			refTargets.push_back(id);
			refEdgeVisited.push_back(edgeVisited);
		}

#ifdef MI_USE_OMP
	} else {
		// run concurrently
		// #pragma omp parallel for private(iter, id, edge_visited_count, tmpTable) shared(refTable, refTargets, refEdgeVisited)

		#pragma omp parallel for ordered
		for (int iter = 0; iter < num_iter; ++iter) {
			int id = cascade.GenRandomNode();
			int edgeVisited;
			vector<RRVec> tmpTable;
			cascade.ReversePropagate(1, id, tmpTable, edgeVisited);
			#pragma omp critical
			{
				refTable.push_back(tmpTable[0]);
				refTargets.push_back(id);
				refEdgeVisited.push_back(edgeVisited);
			}
		}
	}
#endif

}

void RRInflBase::_RebuildRRIndices()
{
	degrees.clear();
	degrees.resize(n, 0);
	degreeRRIndices.clear();

	for (int i = 0; i < n; ++i) {
		degreeRRIndices.push_back( vector<int>() );
	}
	
	// to count hyper edges:
	for (size_t i = 0; i < table.size(); ++i) {
		const RRVec& RR = table[i];
		for (int source : RR) {
			degrees[source]++;
			degreeRRIndices[source].push_back(i); // add index of table
		}
	}

	// add to sourceSet where node's degree > 0
	sourceSet.clear();
	for (size_t i = 0; i < degrees.size(); ++i) {
		if (degrees[i] >= 0) {
			sourceSet.insert(i);
		}
	}
}


// Apply Greedy to solve Max Cover
double RRInflBase::_RunGreedy(int seed_size, 
						vector<int>& outSeeds,
						vector<double>& outEstSpread) 
{
	outSeeds.clear();
	outEstSpread.clear();

	// set enables for table
	vector<bool> enables;
	enables.resize(table.size(), true);

	set<int> candidates(sourceSet);
	CountComparator comp(degrees);

	double spread = 0;
	for (int iter = 0; iter < seed_size; ++iter) {
		set<int>::const_iterator maxPt = max_element(candidates.begin(), candidates.end(), comp);
		int maxSource = *maxPt;
		// cout << "" << maxSource << "\t";
		assert(degrees[maxSource] >= 0);

		// selected one node
		outSeeds.push_back(maxSource);

		// estimate spread
		spread = spread + ((double) n * degrees[maxSource] / table.size() );

		// if (iter==0)
		// {
		// 	  cout << "hhh" << maxSource << (double)n * degrees[28] / table.size() << "hhh" << endl;
		// }

		outEstSpread.push_back(spread);

		// clear values
		candidates.erase(maxPt);
		degrees[maxSource] = -1;

		// deduct the counts from the rest nodes
		const vector<int>& idxList = degreeRRIndices[maxSource];
		if (!isConcurrent) {
			for (int idx : idxList) {
				if (enables[idx]) {
					const RRVec& RRset = table[idx];
					for (int rr : RRset) {
						if (rr == maxSource) continue;
						degrees[rr]--; // deduct
					}
					enables[idx] = false;
				}
			}
		} else {
			// run concurrently
			#pragma omp parallel for
			for (int idxIter = 0; idxIter < idxList.size(); ++idxIter) {
				int idx = idxList[idxIter];
				if (enables[idx]) {
					const RRVec& RRset = table[idx];
					for (int rr : RRset) {
						if (rr == maxSource) continue;
						
						#pragma omp atomic
						degrees[rr]--; // deduct
					}
					enables[idx] = false;
				}
			}
		}
	}
    
	assert(outSeeds.size() == seed_size);
	assert(outEstSpread.size() == seed_size);
	
	return spread;
}

///activate function
double CIMM::function(double x){
	return 2 * x - x * x;
}

// Continuous Greedy to solve CIMM
double CIMM::_RunGreedy(int budget_size,
	double stepsize,
	vector<double>& budget_allocation)
{
	budget_allocation.resize(n, 0.0);

	double spread = 0;
	vector<double> s;
	s.resize(table.size(), 1.0);
	vector<double> delta;
	delta.resize(n, 0.0);

	for (int iter = 0; iter <= (budget_size / stepsize-1); ++iter) {
		for (vector<double>::iterator it = begin(delta);  it < end(delta);  it++)
		{
			*it = 0.0;
		}
		for (int v = 0; v < n; ++v){
			if (budget_allocation[v] + stepsize <= 1.0)
			{
				vector<int> idxList = degreeRRIndices[v];
				for (int i : idxList)
				{
					delta[v] = delta[v] + s[i] * (function(budget_allocation[v] + stepsize) - function(budget_allocation[v])) / (function(1 - budget_allocation[v]));
				}
			}
		}

		vector<double>::iterator max, it;
		max = begin(delta);
		for (it = begin(delta)+1; it < end(delta); it++)
		{
			if (*max <= *it)
				max = it;
		}

		int w = distance(begin(delta), max);

		spread = spread + (double)n* delta[w] / table.size();
		vector<int> idxList = degreeRRIndices[w];
		for (int i : idxList)
		{
			s[i] = s[i] * (1 - function(budget_allocation[w] + stepsize)) / (1 - function(budget_allocation[w]));
			if (s[i] < 0 || s[i] > 1){
				cout << s[i] << endl;
			}
		}
		budget_allocation[w] = budget_allocation[w] + stepsize;
	}

	return spread;
}

vector<int> vectors_intersection(vector<int> v1, vector<int> v2){
	vector<int> v;
	sort(v1.begin(), v1.end());
	sort(v2.begin(), v2.end());
	set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(), back_inserter(v));//求交集   
	return v;
}


/*double CIMM::_RunGreedy_improved(int budget_size,
	double stepsize,
	vector<double>& budget_allocation)
{
	budget_allocation.resize(n, 0.0);

	double spread = 0;
	vector<double> s;
	s.resize(table.size(), 1.0);
	vector<double> delta;
	delta.resize(n, 0.0);
	vector<double> ddelta;
	ddelta.resize(n, 0.0);

	for (int v = 0; v < n; ++v)
	{
		vector<int> idxList = degreeRRIndices[v];
		for (int i : idxList){
			delta[v] = delta[v] + function(stepsize);
		}
	}
	vector<double>::iterator max, it;
	max = begin(delta);
	for (it = begin(delta) + 1; it < end(delta); it++)
	{
		if (*max <= *it)
			max = it;
	}
	int w = distance(begin(delta), max);
	spread = spread + (double)n*delta[w] / table.size();

	vector<int> idxList = degreeRRIndices[w];
	for (int i : idxList)
	{
		s[i] = 1 - function(stepsize);
		assert(s[i] >= 0 && s[i] <= 1);
	}
	budget_allocation[w] = budget_allocation[w] + stepsize;

	for (int iter = 1; iter <= (budget_size / stepsize - 1); ++iter) 
	{
		for (int v = 0; v < n; v++)
		{
			if (budget_allocation[v] + stepsize <= 1.0){
				if (v != w)
				{
					vector<int> idxListt = vectors_intersection(degreeRRIndices[v], idxList);
					for (int i : idxListt)
					{
						ddelta[v] = ddelta[v] + s[i] * (function(budget_allocation[v] + stepsize) - function(budget_allocation[v])) / (1 - budget_allocation[v]) * (function(budget_allocation[w]) - function(budget_allocation[w] - stepsize)) / (1 - function(budget_allocation[w] - stepsize));
					}
				}
				else
				{
					for (int i : idxList)
					{
						ddelta[v] = ddelta[v] + s[i] * (2 * function(budget_allocation[w]) - function(budget_allocation[w] - stepsize) - function(budget_allocation[w] + stepsize)) / (1 - function(budget_allocation[w] - stepsize));
					}
				}
				delta[v] = delta[v] - ddelta[v];
			}
			else
			{
				delta[v] = 0;
			}
		}
		
		max = begin(delta);
		for (it = begin(delta) + 1; it < end(delta); it++)
		{
			if (*max <= *it)
				max = it;
		}

		w = distance(begin(delta), max);
		spread = spread + (double)n* delta[w] / table.size();

		idxList = degreeRRIndices[w];
		for (int i : idxList)
		{
			s[i] = s[i] * (1 - function(budget_allocation[w] + stepsize)) / (1 - function(budget_allocation[w]));
			assert(s[i] >= 0 && s[i] <= 1);
		}
		budget_allocation[w] = budget_allocation[w] + stepsize;
		for (vector<double>::iterator it = begin(ddelta); it < end(ddelta); it++)
		{
			*it = 0.0;
		}
	}
	int sum = 0;
	for (int i = 0; i < n; i++)
	{
		sum = sum + budget_allocation[i];
	}
	cout << "sum" << sum << endl;
	return spread;
}*/
















void RRInflBase::_SetResults(const vector<int>& seeds, const vector<double>& cumu_spread)
{
	for (int i = 0; i < top; ++i) {
		list[i] = seeds[i];
		// d[i] = cumu_spread[i];  // use cumulative spread
		d[i] = (i > 0) ? (cumu_spread[i] - cumu_spread[i - 1]) : cumu_spread[i]; // use marginal spread
	}
}

// Same thing has been implemented as a part of _Greedy
double RRInflBase::_EstimateInfl(const vector<int>& seeds, vector<double>& out_cumu_spread)
{
	set<int> covered;
	double spd = 0;

	vector<bool> enables;
	enables.resize(table.size(), true);
	for (size_t i = 0; i < seeds.size(); ++i) {
		int sd = seeds[i];
		for (int idx : degreeRRIndices[sd]) {
			if (enables[idx]) {
				covered.insert(idx);
				enables[idx] = false;
			}
		}
		spd = (double)(n * covered.size()) / table.size();
		out_cumu_spread.push_back(spd);
	}
	return spd;
}

// Same thing has been implemented as a part of c_Greedy
double CIMM::_EstimateInfl(vector<double> budget)
{
	double spread = 0;
	for (int i = 0; i < table.size(); i++)
	{
		double prod = 1;
		const RRVec& RRset = table[i];
		for (int j : RRset)
		{
			prod = prod * (1 - function(budget[j]));
		}
		spread = spread + 1 - prod;
	}
	return n*spread/table.size();
}


////////////////////////////////////////////////////////////////////
// RRInfl: paper [1]
double RRInfl::DefaultRounds(int n, int m, double epsilon)
{
	return max(144.0 * (n + m) / pow(epsilon, 3) * log( max(n,1) ), 1.0); // to make it positive
}

void RRInfl::BuildInError(graph_type& gf, int k, cascade_type& cascade, double epsilon/*=0.1*/)
{
	n = gf.GetN();
	m = gf.GetM();
	size_t num_iter = (size_t)(ceil(DefaultRounds(n, m, epsilon)));
	_Build(gf, k, cascade, num_iter);
}

void RRInfl::Build(graph_type& gf, int k, cascade_type& cascade, size_t num_iter/*=1000000*/)
{
	n = gf.GetN();
	m = gf.GetM();
	_Build(gf, k, cascade, num_iter);
}
	

void RRInfl::_Build(graph_type& gf, int k, cascade_type& cascade, size_t num_iter)
{
	InitializeConcurrent();

	n = gf.GetN();
	m = gf.GetM();

	top = k;
	d.resize(top, 0.0);
	list.resize(top, 0);
	cascade.Build(gf);

	cout << "#round = " << num_iter << endl;

	// table: target_id  RRset
	// id1  RR: rr1, rr2 ...
	// id2  RR: rr1, rr2 ...
	// ...
	table.clear();
	targets.clear();

	EventTimer pctimer;
	pctimer.SetTimeEvent("start");

	// Step 1:
	_AddRRSimulation(num_iter, cascade, table, targets);
	assert(targets.size() == num_iter);
	assert(table.size() == num_iter);
	pctimer.SetTimeEvent("step1");

	// Step 2:
	vector<int> seeds;
	vector<double> est_spread;
	_RebuildRRIndices();
	pctimer.SetTimeEvent("step2");

	// Step 3:
	double spd = _RunGreedy(k, seeds, est_spread);
	_SetResults(seeds, est_spread);
	pctimer.SetTimeEvent("end");

	cout << "  final (estimated) spread = " << spd << "\t round = " << num_iter << endl;
	
	// Write results to file:
	//FILE *out;
	//fopen_s(&out, file.c_str(), "w");
	//fprintf(out, "%d\n", top);
	//for (int i=0; i<top; i++)
	//	fprintf(out, "%d\t%g\n", list[i], d[i]);
	//fclose(out);
	WriteToFile(file, gf);

	FILE *timetmpfile;
	fopen_s(&timetmpfile, time_file.c_str(), "w");
	fprintf(timetmpfile,"%g\n", pctimer.TimeSpan("start", "end"));
	fprintf(timetmpfile,"Gen graph: %g\n", pctimer.TimeSpan("start", "step1"));
	fprintf(timetmpfile,"Build RR: %g\n", pctimer.TimeSpan("step1", "step2"));
	fprintf(timetmpfile,"Greedy: %g\n", pctimer.TimeSpan("step2", "end"));
	fclose(timetmpfile);
}




///////////////////////////////////////////////////////////////////////////
/// TimPlus: paper 2
void TimPlus::Build(graph_type& gf, int k, cascade_type& cascade, double eps/*=0.1*/, double ell/*=1.0*/)
{
	InitializeConcurrent();

	n = gf.GetN();
	m = gf.GetM();
	top = k;
	d.resize(top, 0.0);
	list.resize(top, 0);

	ell = ell + log(2) / log(n); // Original IMM has failure probability 2/n^ell, so use this transformation to 
	// make the failure probability 1/n^ell

	cascade.Build(gf);
	table.clear();
	targets.clear();

	EventTimer pctimer;
	EventTimer pctimer2;
	pctimer.SetTimeEvent("start");

	// Step1: 
	double est_opt1 = 0.0;
	size_t round1 = 0;
	{
		cout << "Step 1: estimate EPT" << endl;
		pctimer2.SetTimeEvent("est_start");
		int maxIth = (int) ceil(log(n) / log(2) - 1);
		for (int ith=1; ith < maxIth; ++ith) {
			double lb = pow((double)0.5, ith);

			int loop = (int)(StepThreshold(n, lb, ell) + 1);

			vector<int> edgeVisited;
			_AddRRSimulation(loop, cascade, table, targets, edgeVisited);
			round1 += loop;

			double sumKappa = 0.0;
			for (int visit : edgeVisited) {
				double width = (double)visit / m;
				double kappa = 1.0 - pow(1.0 - width, k);
				sumKappa += kappa;
			}
			if (sumKappa / loop > lb) {
				// Algorithm 2 in [2] uses  (n * sumKappa) / (2.0 * loop), but their code does not have 2.0
				est_opt1 = (n * sumKappa) / (loop); 
				break;
			}
		}
		est_opt1 = max(est_opt1, (double)k);
		pctimer2.SetTimeEvent("est_end");
	}
	cout << "  est_opt1 = " << est_opt1 << "\t round1 = " << round1 << endl;
	pctimer.SetTimeEvent("step1");
	double time_per_round = pctimer2.TimeSpan("est_start", "est_end") / ((double) round1);
	cout << "  (Time per round = " << time_per_round << ")" << endl;

	// Step2: Use greedy to estimate opt lower bound
	double eps_step2, spd2, est_opt2;
	size_t round2;
	{
		eps_step2 = EpsPrime(eps, k, ell);
		vector<int> seeds2;
		vector<double> est_spread2;
		double theta = RThreshold_0(eps, est_opt1, ell);
		round2 = (size_t)max(theta+1.0, 1.0);
		cout << "Step 2: estimate opt by greedy. round = " << round2 << endl
			 <<  "  (Estimate time = " << time_per_round * round2 << ")"  << endl;
		_AddRRSimulation(round2, cascade, table, targets);
		_RebuildRRIndices();
		spd2 = _RunGreedy(k, seeds2, est_spread2);
		est_opt2 = spd2 / (1+eps_step2);
		est_opt2 = max(est_opt2, est_opt1);
	}
	cout << "  est_opt2 = " << est_opt2 << "\t round2 = " << round2 << endl;
	pctimer.SetTimeEvent("step2");

	// Step3: Set final runs
	vector<int> seeds3;
	vector<double> est_spread3;
	double spd3;
	size_t round3;
	{
		double theta = RThreshold(eps, est_opt2, k, ell);
		round3 = (size_t)max(theta+1.0, 1.0);
		cout << "Step 3: final greedy. round = " << round3 << endl
			 <<  "  (Estimate time = " << time_per_round * round3 << ")" << endl;
		pctimer.SetTimeEvent("step3-1");
		_AddRRSimulation(round3, cascade, table, targets);
		pctimer.SetTimeEvent("step3-2");
		_RebuildRRIndices();
		pctimer.SetTimeEvent("step3-3");
		spd3 = _RunGreedy(top, seeds3, est_spread3);
	}
	cout << "  final spread = " << spd3 << "\t round3 = " << round3 << endl;
	_SetResults(seeds3, est_spread3);
	pctimer.SetTimeEvent("end");

	// Write results to file:
	//FILE *out;
	//fopen_s(&out, file.c_str(), "w");
	//fprintf(out, "%d\n", top);
	//for (int i=0; i<top; i++)
	//	fprintf(out, "%d\t%g\n", list[i], d[i]);
	//fclose(out);
	WriteToFile(file, gf);

	FILE *timetmpfile;
	fopen_s(&timetmpfile, time_file.c_str(), "w");
	fprintf(timetmpfile,"%g\n", pctimer.TimeSpan("start", "end"));
	fprintf(timetmpfile,"Step1: %g\n", pctimer.TimeSpan("start", "step1"));
	fprintf(timetmpfile,"Step2: %g\n", pctimer.TimeSpan("step1", "step2"));
	fprintf(timetmpfile,"Step3: %g\n", pctimer.TimeSpan("step2", "end"));
	fprintf(timetmpfile,"   Step3-1: %g\n", pctimer.TimeSpan("step3-1", "step3-2"));
	fprintf(timetmpfile,"   Step3-2: %g\n", pctimer.TimeSpan("step3-2", "step3-3"));
	fprintf(timetmpfile,"   Step3-3: %g\n", pctimer.TimeSpan("step3-3", "end"));
	fclose(timetmpfile);
}

double TimPlus::LogNChooseK(int n, int k){
	double ans = 0.0;
	assert(n >= k && k >= 0);
	for(int i = 0; i < k; i++){
		ans += log(n-i);
	}
	for(int i = 1; i <= k; i++){
		ans -= log(i);
	}
	return ans;
}

double TimPlus::RThreshold_0(double eps, double opt, double ell/*=1.0*/)
{
	double lambda = (double)(8 + 2*eps) * n * (ell*log(n) + log(2)) / (eps*eps);
	double theta = lambda / (opt*4.0);
	return theta;
}

double TimPlus::RThreshold(double eps, double opt, int k, double ell/*=1.0*/)
{
	double lambda = (double)(8 + 2*eps) * n * (ell*log(n) + LogNChooseK(n,k) + log(2)) / (eps*eps);
	double theta = lambda / opt;
	return theta;
}


double TimPlus::EpsPrime(double eps, int k, double ell/*=1.0*/) 
{
	return 5.0 * pow(eps*eps*ell/((double)ell+k), 1.0/3.0);
}

double TimPlus::StepThreshold(int n, double lb, double ell/*=1.0*/)
{
	double logValue = max(log(n) / log(2.0), 1.0);
	double loop = ((double)6 * ell * log(n) + 6 * log(logValue)) / lb;
	return loop;
}


///////////////////////////////////////////////////////////////////////////
/// IMM: paper 3
/// mode = 0, the original IMM algorithm
///      = 1, the first workaround fixing the IMM issue reported in arXiv:1808.09363
///      = 2, the second workaround fixing the IMM issue reported in arXiv:1808.09363
void IMM::Build(graph_type& gf, int k, cascade_type& cascade, double eps /* = 0.1 */, double ell /* = 1.0 */, int mode /* = 0 */)
{
	InitializeConcurrent();

	EventTimer pctimer;
	vector<EventTimer> steptimers;
	pctimer.SetTimeEvent("start");

	n = gf.GetN();
	m = gf.GetM();
	top = k;
	d.resize(top, 0.0);
	list.resize(top, 0);

	ell = ell + log(2) / log(n); // Original IMM has failure probability 2/n^ell, so use this transformation to 
								// make the failure probability 1/n^ell

	cascade.Build(gf);

	table.clear();
	targets.clear();

	double epsprime = eps * sqrt(2.0); // eps'
	double LB = 1.0;   // lower bound
	double maxRounds = max(max(log2((double)n), 1.0) - 1.0, 1.0);

	if (mode == 2) {
		// second workaround for IMM, see arXiv:1808.09363
		double gamma, lgamma, rgamma;

		lgamma = 0;
		rgamma = 2;
		

		// doubling search to find the correct right-side gamma;
		while (ceil(LambdaStar(eps, k, ell + rgamma, n)) > pow(n,rgamma)) {
			lgamma = rgamma;
			rgamma = rgamma * 2;
		}

		// binary search to find the gamma in the middle;
		while (rgamma > lgamma + 0.1) {
			gamma = (lgamma + rgamma) / 2;
			if (ceil(LambdaStar(eps, k, ell + gamma, n)) > pow(n, gamma)) {
				lgamma = gamma;
			}
			else {
				rgamma = gamma;
			}
		}

		ell = ell + rgamma;
		cout << "  IMM Workaround 2: gamma = " << rgamma << endl;
	}

	// first a few passes of sampling
	double spread = 0.0;
	for (int r = 1; r < maxRounds; r++) {
		EventTimer stept;
		stept.SetTimeEvent("step_start");

		cout << "  Step" << r << ":" <<endl;
		double x = max(double(n) / pow(2, r), 1.0);
		double theta = LambdaPrime(epsprime, k, ell, n) / x;

		// select nodes
		vector<int> seeds;
		vector<double> est_spread;
		LARGE_INT64 nNewSamples = LARGE_INT64(theta - table.size() + 1);
		if (table.size() < theta) {
			// generate samples
			_AddRRSimulation(nNewSamples, cascade, table, targets);
			_RebuildRRIndices();
			spread = _RunGreedy(top, seeds, est_spread);
			_SetResults(seeds, est_spread);
		}
		cout << " spread = " << spread << "\t round = " << ((nNewSamples > 0) ? nNewSamples : 0) << endl;

		stept.SetTimeEvent("step_end");
		steptimers.push_back(stept);

		// check influence and set lower bound
		if (spread >= ((1.0 + epsprime) * x)) {
			LB = spread / (1.0 + epsprime);
			break;
		}
	}

	// final pass of sampling
	{
		EventTimer stept;
		stept.SetTimeEvent("step_start");
		cout << "  Estimated Lower bound: " << LB << endl;
		cout << "  Final Step:" << endl;
		double theta = LambdaStar(eps, k, ell, n) / LB;

		vector<int> seeds;
		vector<double> est_spread;



		if (mode == 1) {
			LARGE_INT64 nNewSamples = LARGE_INT64(theta + 1);
			cout << "  IMM Workaround 1 --- Regenerating RR sets, # RR sets = " << nNewSamples  << endl;
			table.clear();
			targets.clear();
			// generate samples
			_AddRRSimulation(nNewSamples, cascade, table, targets);
			_RebuildRRIndices();
			spread = _RunGreedy(top, seeds, est_spread);
			_SetResults(seeds, est_spread);

		}
		else {
			LARGE_INT64 nNewSamples = LARGE_INT64(theta - table.size() + 1);
			nNewSamples = ((nNewSamples > 0) ? nNewSamples : 0);
			cout << "  Original IMM without regenerating RR sets, # new RR sets needed = " << nNewSamples << endl;
			if (table.size() < theta) {
				// generate samples
				_AddRRSimulation(nNewSamples, cascade, table, targets);
				_RebuildRRIndices();
				spread = _RunGreedy(top, seeds, est_spread);
				_SetResults(seeds, est_spread);
			}
		}


		cout << " spread = " << spread << endl;


		stept.SetTimeEvent("step_end");
		steptimers.push_back(stept);
	}
	pctimer.SetTimeEvent("end");

	// Write results to file:
	//FILE *out;
	//fopen_s(&out, file.c_str(), "w");
	//fprintf(out, "%d\n", top);
	//for (int i = 0; i<top; i++)
	//	fprintf(out, "%d\t%g\n", list[i], d[i]);
	//fclose(out);

	WriteToFile(file, gf);

	FILE *timetmpfile;
	fopen_s(&timetmpfile, time_file.c_str(), "w");
	fprintf(timetmpfile, "%g\n", pctimer.TimeSpan("start", "end"));
	for (size_t t = 0; t < steptimers.size(); t++) {
		if (t != steptimers.size() - 1) {
			fprintf(timetmpfile, "  Step%d: %g\n", t + 1, steptimers[t].TimeSpan("step_start", "step_end"));
		}
		else {
			fprintf(timetmpfile, "  Final Step: %g\n", steptimers[t].TimeSpan("step_start", "step_end"));
		}
	}
	fclose(timetmpfile);
}

double IMM::LambdaPrime(double epsprime, int k, double ell, int n)
{
	static double SMALL_DOUBLE = 1e-16;
	double cst = (2.0 + 2.0 / 3.0 * epsprime) / max(epsprime * epsprime, SMALL_DOUBLE);
	double part2 = LogNChooseK(n, k);
	part2 += ell * log(max((double)n, 1.0));
	part2 += log(max(log2(max((double)n, 1.0)), 1.0));
	// calc \lambda'
	double lambda = cst * part2 * n;
	return lambda;
}

double IMM::LambdaStar(double eps, int k, double ell, int n)
{
	static double SMALL_DOUBLE = 1e-16;
	static double APPRO_RATIO = (1.0 - 1.0 / exp(1));
	double logsum = ell * log(n) + log(2);
	// calc \alpha and \beta
	double alpha = sqrt(max(logsum, SMALL_DOUBLE));
	double beta = sqrt(APPRO_RATIO * (LogNChooseK(n, k) + logsum));
	// calc \lambda*
	double lambda = 2.0 * n / max(pow(eps, 2), SMALL_DOUBLE);
	lambda = lambda * pow(APPRO_RATIO * alpha + beta, 2);
	return lambda;
}


///CIMM: 
void CIMM::Build(graph_type& gf, int k, cascade_type& cascade, double eps /* = 0.1 */, double ell /* = 1.0 */, double delta /* = 1.0*/)
{
	InitializeConcurrent();

	EventTimer pctimer;
	vector<EventTimer> steptimers;
	pctimer.SetTimeEvent("start");

	n = gf.GetN();
	m = gf.GetM();
	top = k;
	d.resize(top, 0.0);
	list.resize(top, 0);

	ell = ell + log(2) / log(n); // Original IMM has failure probability 2/n^ell, so use this transformation to 
	// make the failure probability 1/n^ell

	cascade.Build(gf);

	table.clear();
	targets.clear();

	double epsprime = eps * sqrt(2.0); // eps'
	double LB = 1.0;   // lower bound
	double maxRounds = max(max(log2((double)n), 1.0) - 1.0, 1.0);

    vector<double> cimm_budget;
	cimm_budget.resize(n, 0.0);
	// first a few passes of sampling
	double spread = 0.0;
	for (int r = 1; r < maxRounds; r++) {
		EventTimer stept;
		stept.SetTimeEvent("step_start");

		cout << "  Step" << r << ":";
		double x = max(double(n) / pow(2, r), 1.0);
		double theta = LambdaPrime(epsprime, k, ell, n) / x;

		// select nodes
		vector<double> budget;

		LARGE_INT64 nNewSamples = LARGE_INT64(theta - table.size() + 1);
		if (table.size() < theta) {
			// generate samples
			_AddRRSimulation(nNewSamples, cascade, table, targets);
			_RebuildRRIndices();
			spread = _RunGreedy(top, delta, budget);
			cout << _EstimateInfl(budget) << endl;
			for (int i = 0; i < n; i++)
			{
				cimm_budget[i] = budget[i];
			}
		}
		
		cout << " spread = " << spread << "\t round = " << ((nNewSamples > 0) ? nNewSamples : 0) << endl;

		stept.SetTimeEvent("step_end");
		steptimers.push_back(stept);

		// check influence and set lower bound
		if (spread >= ((1.0 + epsprime) * x)) {
			LB = spread / (1.0 + epsprime);
			break;
		}
	}

	// final pass of sampling
	{
		EventTimer stept;
		stept.SetTimeEvent("step_start");
		cout << "  Estimated Lower bound: " << LB << endl;
		cout << "  Final Step:";
		double theta = LambdaStar(eps, k, ell, n) / LB;
		vector<double> budget;

		LARGE_INT64 nNewSamples = LARGE_INT64(theta - table.size() + 1);
		if (table.size() < theta) {
			// generate samples
			_AddRRSimulation(nNewSamples, cascade, table, targets);
			_RebuildRRIndices();
			spread = _RunGreedy(top, delta, budget);

			for (int i = 0; i < n; i++)
			{
				cimm_budget[i] = budget[i];
			}
		}

		cout << " spread = " << spread << "\t round = " << ((nNewSamples > 0) ? nNewSamples : 0) << endl;


		stept.SetTimeEvent("step_end");
		steptimers.push_back(stept);
	}
	pctimer.SetTimeEvent("end");

	FILE *tmpfile;
	fopen_s(&tmpfile, file.c_str(), "w");
	fprintf(tmpfile, "%g\n", spread);
	for (int i = 0; i < n; i++){
		fprintf(tmpfile, "Node%d: %g\n", i, cimm_budget[i]);///actually here i is not the node id... but the expected influence is accurate
	}
	fclose(tmpfile);

	FILE *timetmpfile;
	fopen_s(&timetmpfile, time_file.c_str(), "w");
	fprintf(timetmpfile, "%g\n", pctimer.TimeSpan("start", "end"));
	for (size_t t = 0; t < steptimers.size(); t++) {
		if (t != steptimers.size() - 1) {
			fprintf(timetmpfile, "  Step%d: %g\n", t + 1, steptimers[t].TimeSpan("step_start", "step_end"));
		}
		else {
			fprintf(timetmpfile, "  Final Step: %g\n", steptimers[t].TimeSpan("step_start", "step_end"));
		}
	}
	fclose(timetmpfile);
}

void ShapleyInfl::Shapley_Build(graph_type& gf, cascade_type& cascade, GeneralCascade & gc, double eps/*=0.1*/, double ell/*=1.0*/, int topk /*=50*/, bool isSingleInf /* = false */)
{

	InitializeConcurrent();

	EventTimer pctimer;
	EventTimer pctimer2;
	vector<EventTimer> steptimers;

	pctimer.SetTimeEvent("start");

	n = gf.GetN();
	m = gf.GetM();

	std::vector<double> shapleyV;
	std::vector<int> hitCount;
	LARGE_INT64 totalEdgeVisited = 0;
	LARGE_INT64 totalRRSetSoFar = 0; // the number of RR sets already generated

	double shpv = 0.0;

	shapleyV.assign(n, 0.0);
	hitCount.assign(n, 0);

	shapleyValueId nullSVI;
	nullSVI.sid = 0;
	nullSVI.value = 0.0;


	top = 1;  // for legacy reason

	cascade.Build(gf);
	table.clear();
	targets.clear();

	// when n is too small, need to use a smaller topk. A topk that is equal to n is also not good,
	// violating the assumption that \phi_k \ge 1. Thus use n/2. Only for very small graph.
	topk = min(n / 2, topk);

	// Phase 1: Estimate the number of RR sets needed
	cout << "Phase 1: Estimate the number of RR sets needed" << endl;
	pctimer2.SetTimeEvent("phase1_start");
	double LB = 1.0;   // lower bound

	// use the IMM sampling method to estimate lower bound LB
	cout << "lower bound LB estimated using the sampling method similar to IMM" << endl;
	double epsprime = eps * sqrt(2.0); // eps'
	double maxRounds = max(max(log2(n), 1.0) - 1.0, 1.0);

	// first a few passes of sampling, based on the IMM sampling method
	for (int r = 1; r < maxRounds; r++) {
		EventTimer stept;
		stept.SetTimeEvent("step_start");

		cout << "  Step" << r << ":";
		double x = max(double(n) / pow(2, r), 1.0);
		LARGE_INT64 theta = (LARGE_INT64)ceil(n*((ell + 1.0)* log(n) + log(log2(n)) + log(2))* (2.0 + 2.0 / 3.0 * epsprime) / (epsprime * epsprime * x));

		LARGE_INT64 nNewSamples = (LARGE_INT64)(theta - totalRRSetSoFar);

		if (totalRRSetSoFar < theta) {
			// generate samples
			Shapley_AddRRSimulation(nNewSamples, cascade, shapleyV, hitCount, table, totalEdgeVisited);
			totalRRSetSoFar = theta;

			// find the k-th largest Shapley value or SNI value
			shapleyVId.assign(n, nullSVI);

			if (!isSingleInf) {
				for (int i = 0; i < n; i++) {
					shapleyVId[i].sid = i;
					shapleyVId[i].value = shapleyV[i];
				}
			}
			else { /* if computing single node influence, use the shapleyVId array, but use hitCount collected from the simulation */
				for (int i = 0; i < n; i++) {
					shapleyVId[i].sid = i;
					shapleyVId[i].value = (double) hitCount[i]; 
				}
			}

			std::sort(shapleyVId.begin(), shapleyVId.end(), ShapleyComparator());

			shpv = 1.0 * shapleyVId[topk - 1].value / totalRRSetSoFar * n;
		}

		if (!isSingleInf) {
			cout << "No. " << topk << " largest Shapley value = " << shpv << "\t current round = " << ((nNewSamples > 0) ? nNewSamples : 0) << "\t total round = " << totalRRSetSoFar << endl;
		}
		else {
			cout << "No. " << topk << " largest SNI value = " << shpv << "\t current round = " << ((nNewSamples > 0) ? nNewSamples : 0) << "\t total round = " << totalRRSetSoFar << endl;
		}
		stept.SetTimeEvent("step_end");
		steptimers.push_back(stept);

		// check stopping condition and set lower bound
		if (shpv >= ((1.0 + epsprime) * x)) {
			LB = shpv / (1.0 + epsprime);
			break;
		}
	}


	LARGE_INT64 stheta = (LARGE_INT64)ceil(n*((ell + 1.0)* log(n) + log(4))* (2.0 + 2.0 / 3.0 * eps) / (eps * eps * LB));

	pctimer2.SetTimeEvent("phase1_end");
	if (!isSingleInf) {
		cout << "Phase 1 ends: Estimated lower bound of the k-th largest Shapley value: " << LB << endl;
	}
	else {
		cout << "Phase 1 ends: Estimated lower bound of the k-th largest SNI value: " << LB << endl;
	}
	cout << "              Number of RR sets needed for Phase 2: " << stheta << endl;

	// pctimer.SetTimeEvent("step1");

	// Phase 2: generate RR sets and estimate Shapley values of nodes
	// need to start from the scratch, because otherwise the estimates could be unbiased.
	// Martingale property cannot guarantee unbiasness

	if (!isSingleInf) {
		cout << "Phase 2: generate RR sets and estimate Shapley values of all nodes, theta = " << stheta << endl;
	}
	else {
		cout << "Phase 2: generate RR sets and estimate SNI values of all nodes, theta = " << stheta << endl;
	}

	shapleyV.assign(n, 0.0);
	hitCount.assign(n, 0);
	totalEdgeVisited = 0;

	Shapley_AddRRSimulation(stheta, cascade, shapleyV, hitCount, table, totalEdgeVisited);

	if (!isSingleInf) {
		for (int i = 0; i < n; i++) {
			shapleyV[i] = 1.0 * shapleyV[i] * n / stheta;
		}
	}
	else {
		for (int i = 0; i < n; i++) {
			shapleyV[i] = 1.0 * hitCount[i] * n / stheta;
		}
	}


	// pctimer.SetTimeEvent("step2");
	pctimer2.SetTimeEvent("phase2_end");

	double EPT = 1.0 * totalEdgeVisited / stheta;

	cout << "Phase 2 ends" << endl;


	// sort Shapley values. Since we need to sort node id with the value, we need another structure for this

	shapleyVId.assign(n, nullSVI);

	for (int i = 0; i < n; i++) {
		shapleyVId[i].sid = i;
		shapleyVId[i].value = shapleyV[i];
	}
	
	std::sort(shapleyVId.begin(), shapleyVId.end(), ShapleyComparator());

	if (!isSingleInf) {
		cout << "k-th largest Shapley value = " << shapleyVId[topk - 1].value << "; EPT = " << EPT << endl;
	}
	else {
		cout << "k-th largest SNI value = " << shapleyVId[topk - 1].value << "; EPT = " << EPT << endl;
	}

	// this is to indicate that we have found n seeds, from the Shapley value, to be used
	// later for simulation
	top = n; 

	d.resize(top, 0.0);
	list.resize(top, 0);
	// fill list[] and d[] arrays
	for (int i = 0; i < n; i++) {
		list[i] = shapleyVId[i].sid;
		d[i] = shapleyVId[i].value;
	}

	

	// Write results to file:
	FILE *out;
	int nshapley = n;
	fopen_s(&out, file.c_str(), "w");
	fprintf(out, "%d\n", nshapley);
	if (!isSingleInf) {
		fprintf(out, "# nodeid\t Shapley value\t  epsilon = %lg; ell = %lg; EPT = %lg\n", eps, ell, EPT);
	}
	else {
		fprintf(out, "# nodeid\t SNI value\t  epsilon = %lg; ell = %lg; EPT = %lg\n", eps, ell, EPT);
	}
	for (int i = 0; i<nshapley; i++)
		fprintf(out, "%s\t%lg\n", gf.MapIndexToNodeName(list[i]).c_str(), d[i]); 
	fclose(out);


	FILE *timetmpfile;
	fopen_s(&timetmpfile, time_file.c_str(), "w");
	fprintf(timetmpfile, "%lg\n", pctimer2.TimeSpan("phase1_start", "phase2_end"));
	fclose(timetmpfile);
}


void ShapleyInfl::Shapley_AddRRSimulation(LARGE_INT64 num_iter,
	cascade_type& cascade,
	std::vector<double>& shapleyV,
	std::vector<int>& hitCount,  // hitCount[i] is the number of times the RR sets hit node i
	std::vector< std::vector<int> >& refTable,
	LARGE_INT64 & totalEdgeVisited)
	// generate one RR set, and getting the statistics and the estimate of the Shapley values.
{
	for (LARGE_INT64 iter = 0; iter < num_iter; ++iter) {
		int id = cascade.GenRandomNode();
		int edge_visited_count;

		// we do not need the keep the RR sets, so we can disgard them before generating the next one. 
		// I do not want to redefine ReverseGCascade::ReversePropagate, so use clear() here.
		refTable.clear();

		cascade.ReversePropagate(1, id, refTable, edge_visited_count);

		vector<int>& lastRRset = *(refTable.rbegin());
		for (int i = 0; i < lastRRset.size(); i++) {
			shapleyV[lastRRset[i]] += 1.0 / lastRRset.size();
			hitCount[lastRRset[i]] += 1;
		}

		totalEdgeVisited += edge_visited_count;

	}
}

bool ShapleyInfl::ShapleyComparator::operator() (shapleyValueId a, shapleyValueId b) {
	return (a.value > b.value);
}


