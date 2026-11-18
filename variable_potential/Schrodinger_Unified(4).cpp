// Unified adaptation of Reflected.cpp and canonical KNN: free and variable-potential Schrödinger.
// Eigen 3, C++17. Grid indices and PHS reference weights follow the original programs.
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
using Eigen::MatrixXd; using Eigen::VectorXd; using Eigen::VectorXcd; using Eigen::SparseMatrix; using Eigen::Triplet;
constexpr double PI=3.14159265358979323846;
struct Offset {int a,b;};
int idx(int i,int j,int N){return (i-1)*(N-2)+j-1;}
std::vector<Offset> shell(){std::vector<Offset>s;for(int a=-3;a<=3;++a)for(int b=-3;b<=3;++b)if(a*a+b*b<=9)s.push_back({a,b});return s;}
std::vector<Offset> powers(){std::vector<Offset>p;for(int d=0;d<=5;++d)for(int a=0;a<=d;++a)p.push_back({a,d-a});return p;}
double powi(double x,int n){return std::pow(x,n);}
VectorXd phsWeights(const std::vector<Offset>&s){
 auto p=powers();int n=s.size(),np=p.size();MatrixXd A=MatrixXd::Zero(n+np,n+np);VectorXd rhs=VectorXd::Zero(n+np);
 for(int i=0;i<n;++i){double r=std::hypot(double(s[i].a),double(s[i].b));rhs(i)=121*powi(r,9);
  for(int j=0;j<n;++j)A(i,j)=powi(std::hypot(double(s[i].a-s[j].a),double(s[i].b-s[j].b)),11);
  for(int j=0;j<np;++j){double v=powi(double(s[i].a),p[j].a)*powi(double(s[i].b),p[j].b);A(i,n+j)=A(n+j,i)=v;}}
 for(int j=0;j<np;++j)if((p[j].a==2&&p[j].b==0)||(p[j].a==0&&p[j].b==2))rhs(n+j)=2;
 VectorXd scales=A.cwiseAbs().rowwise().maxCoeff();for(int i=0;i<scales.size();++i)if(scales(i)==0)scales(i)=1;
 MatrixXd B=scales.cwiseInverse().asDiagonal()*A;VectorXd b=rhs.cwiseQuotient(scales);
 Eigen::FullPivLU<MatrixXd> lu(B);if(lu.rank()<n+np)throw std::runtime_error("PHS augmented system rank deficient");
 VectorXd sol=lu.solve(b);double res=(A*sol-rhs).norm()/std::max(1.0,rhs.norm());if(res>2e-6)throw std::runtime_error("PHS weight residual exceeds 2e-6: "+std::to_string(res));return sol.head(n);
}
std::pair<int,int> oddFold(int a,int N){int M=N-1;a%=2*M;if(a<0)a+=2*M;if(a==0||a==M)return {0,0};if(a>M)return {2*M-a,-1};return {a,1};}
// Rank-safe distance ordering follows the previous Python matched-KNN pilot.
std::vector<Offset> knnStencil(int i,int j,int N){
 std::vector<std::tuple<int,int,int,int,int>> cand;for(int x=std::max(0,i-10);x<=std::min(N-1,i+10);++x)for(int y=std::max(0,j-10);y<=std::min(N-1,j+10);++y){int a=x-i,b=y-j;if(a*a+b*b<=100)cand.emplace_back(a*a+b*b,x,y,a,b);}
 std::sort(cand.begin(),cand.end());auto p=powers();std::vector<VectorXd>Q;std::vector<Offset>sel;std::map<std::pair<int,int>,bool> used;
 for(const auto&c:cand){int a=std::get<3>(c),b=std::get<4>(c);VectorXd v(p.size());for(int t=0;t<v.size();++t)v(t)=powi(a/5.0,p[t].a)*powi(b/5.0,p[t].b);
  for(const auto&q:Q)v-=q.dot(v)*q;double nv=v.norm();if(nv>1e-10){Q.push_back(v/nv);sel.push_back({a,b});used[{a,b}]=true;if(Q.size()==p.size())break;}}
 if(Q.size()!=p.size())throw std::runtime_error("KNN P5 rank failure");
 for(const auto&c:cand){if(sel.size()==29)break;int a=std::get<3>(c),b=std::get<4>(c);if(!used[{a,b}]){sel.push_back({a,b});used[{a,b}]=true;}}
 if(sel.size()!=29)throw std::runtime_error("KNN stencil too small");return sel;
}
SparseMatrix<double> build(int N,const std::string&method){int n=N-2,sz=n*n;double h=PI/(N-1);std::vector<Triplet<double>> t;t.reserve(size_t(sz)*29);
 if(method=="fd4"){
  int offs[]={-2,-1,0,1,2};double w[]={-1./12,4./3,-2.5,4./3,-1./12};
  for(int i=1;i<N-1;++i)for(int j=1;j<N-1;++j)for(int d=0;d<2;++d)for(int k=0;k<5;++k){auto f=oddFold((d==0?i:j)+offs[k],N);if(f.second==0)continue;int x=d==0?f.first:i,y=d==1?f.first:j;t.emplace_back(idx(i,j,N),idx(x,y,N),f.second*w[k]/(h*h));}
 }else if(method=="reflected"){
  auto s=shell();auto w=phsWeights(s);for(int i=1;i<N-1;++i)for(int j=1;j<N-1;++j)for(int k=0;k<w.size();++k){auto x=oddFold(i+s[k].a,N),y=oddFold(j+s[k].b,N);if(x.second*y.second)t.emplace_back(idx(i,j,N),idx(x.first,y.first,N),x.second*y.second*w(k)/(h*h));}
 }else if(method=="knn"){
  // Cache by relative offset pattern, as in the original matched-KNN experiment.
  std::map<std::vector<std::pair<int,int>>,VectorXd>cache;
  for(int i=1;i<N-1;++i)for(int j=1;j<N-1;++j){auto s=knnStencil(i,j,N);std::vector<std::pair<int,int>>key;for(auto z:s)key.emplace_back(z.a,z.b);
   auto it=cache.find(key);if(it==cache.end())it=cache.emplace(key,phsWeights(s)).first;
   for(int k=0;k<29;++k){int x=i+s[k].a,y=j+s[k].b;if(x>0&&x<N-1&&y>0&&y<N-1)t.emplace_back(idx(i,j,N),idx(x,y,N),it->second(k)/(h*h));}}
 }else throw std::runtime_error("Unknown method: "+method);
 SparseMatrix<double>L(sz,sz);L.setFromTriplets(t.begin(),t.end());L.makeCompressed();return L;
}
double potential(double x,double y){return 5*std::pow(std::sin(x),2)*std::pow(std::sin(y),2)+2*std::sin(x)*std::sin(2*y);}
struct Options{std::string problem="potential",method="reflected",out="result.csv";int N=31,k=2,ell=3;double T=.2,dt=0;};
Options parse(int argc,char**argv){Options o;for(int i=1;i<argc;++i){std::string a=argv[i];if(i+1>=argc)throw std::runtime_error("Missing value for "+a);std::string v=argv[++i];if(a=="--problem")o.problem=v;else if(a=="--method")o.method=v;else if(a=="--N")o.N=std::stoi(v);else if(a=="--T")o.T=std::stod(v);else if(a=="--dt")o.dt=std::stod(v);else if(a=="--k")o.k=std::stoi(v);else if(a=="--ell")o.ell=std::stoi(v);else if(a=="--out")o.out=v;else throw std::runtime_error("Unknown option "+a);}if(o.N<7||o.T<=0)throw std::runtime_error("Invalid N or T");if(o.problem!="free"&&o.problem!="potential")throw std::runtime_error("--problem must be free or potential");return o;}
int main(int argc,char**argv){try{auto o=parse(argc,argv);auto start=std::chrono::steady_clock::now();auto L=build(o.N,o.method);int n=L.rows();double h=PI/(o.N-1);
 VectorXd V=VectorXd::Zero(n);VectorXcd z(n);for(int i=1;i<o.N-1;++i)for(int j=1;j<o.N-1;++j){int p=idx(i,j,o.N);double x=i*h,y=j*h;if(o.problem=="potential"){V(p)=potential(x,y);double amp=std::sin(x)*std::sin(y)*std::exp(-8*std::pow(x-PI/2,2)-8*std::pow(y-PI/2,2));z(p)=std::polar(amp,3*x+2*y);}else z(p)=std::sin(o.k*x)*std::sin(o.ell*y);}
 auto H=[&](const VectorXcd&a)->VectorXcd{return -(L*a).eval()+V.cast<std::complex<double>>().cwiseProduct(a);};
 auto mass=[&](const VectorXcd&a){return a.squaredNorm();};auto energy=[&](const VectorXcd&a){return std::real(a.dot(H(a)));};
 double m0=mass(z),e0=energy(z);double rowmax=0;for(int k=0;k<L.outerSize();++k)for(SparseMatrix<double>::InnerIterator it(L,k);it;++it)rowmax=std::max(rowmax,std::abs(it.value()));
 // Conservative RK4 step bound from a max absolute row sum (Gershgorin).
 VectorXd rows=VectorXd::Zero(n);for(int k=0;k<L.outerSize();++k)for(SparseMatrix<double>::InnerIterator it(L,k);it;++it)rows(it.row())+=std::abs(it.value());
 double bound=rows.maxCoeff()+V.cwiseAbs().maxCoeff();double dt=o.dt>0?o.dt:std::min(0.002,1.5/std::max(bound,1.0));int steps=std::ceil(o.T/dt);dt=o.T/steps;
 std::complex<double> minusI(0,-1);for(int step=0;step<steps;++step){VectorXcd k1=minusI*H(z),k2=minusI*H(z+0.5*dt*k1),k3=minusI*H(z+0.5*dt*k2),k4=minusI*H(z+dt*k3);z+=(dt/6.)*(k1+2*k2+2*k3+k4);}
 double err=std::numeric_limits<double>::quiet_NaN();if(o.problem=="free"){double omega=o.k*o.k+o.ell*o.ell;VectorXcd exact(n);for(int i=1;i<o.N-1;++i)for(int j=1;j<o.N-1;++j)exact(idx(i,j,o.N))=std::polar(std::sin(o.k*i*h)*std::sin(o.ell*j*h),-omega*o.T);err=(z-exact).norm()/exact.norm();}
 SparseMatrix<double>dif=L-SparseMatrix<double>(L.transpose());double sym=dif.norm()/L.norm();double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
 std::ofstream f(o.out);f<<std::setprecision(16)<<"problem,method,N,T,dt,steps,relative_exact_error,mass_drift,energy_drift,symmetry_defect,nnz,seconds\n"<<o.problem<<','<<o.method<<','<<o.N<<','<<o.T<<','<<dt<<','<<steps<<','<<err<<','<<std::abs(mass(z)-m0)/m0<<','<<std::abs(energy(z)-e0)/std::max(std::abs(e0),1e-30)<<','<<sym<<','<<L.nonZeros()<<','<<elapsed<<'\n';
 std::cout<<"problem="<<o.problem<<" method="<<o.method<<" N="<<o.N<<" steps="<<steps<<" exact_error="<<err<<" mass_drift="<<std::abs(mass(z)-m0)/m0<<" symmetry="<<sym<<"\n";
 // Grid solution output supports independent Python reference comparison.
 std::string sol=o.out+".solution.csv";std::ofstream g(sol);g<<std::setprecision(17)<<"i,j,x,y,u,v\n";for(int i=1;i<o.N-1;++i)for(int j=1;j<o.N-1;++j){auto a=z(idx(i,j,o.N));g<<i<<','<<j<<','<<i*h<<','<<j*h<<','<<a.real()<<','<<a.imag()<<'\n';}
 }catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}}
