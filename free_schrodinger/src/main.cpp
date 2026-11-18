#include <boost/multiprecision/cpp_dec_float.hpp>
#include <algorithm>
#include <cmath>
#include <complex>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
using HP=boost::multiprecision::number<boost::multiprecision::cpp_dec_float<65>>;
using namespace std;
const double PI=acos(-1.0);
struct Offset {int a,b;};
vector<Offset> shell(int r2){vector<Offset> s;for(int a=-4;a<=4;a++)for(int b=-4;b<=4;b++)if(a*a+b*b<=r2)s.push_back({a,b});return s;}
vector<pair<int,int>> powers(int m){vector<pair<int,int>> v;for(int d=0;d<=m;d++)for(int a=0;a<=d;a++)v.push_back({a,d-a});return v;}
HP ipow(HP x,int p){HP v=1;for(int i=0;i<p;i++)v*=x;return v;}
HP kernel(HP r2,int q,bool logarithmic){if(r2==0)return 0;using boost::multiprecision::sqrt;using boost::multiprecision::log;HP r=sqrt(r2);HP v=ipow(r,q);return logarithmic?v*log(r):v;}
HP lapkernel(HP r2,int q,bool logarithmic){if(r2==0)return 0;using boost::multiprecision::sqrt;using boost::multiprecision::log;HP r=sqrt(r2);HP v=ipow(r,q-2);return logarithmic?v*(HP(q*q)*log(r)+2*q):v*q*q;}
vector<HP> solve(vector<vector<HP>> a,vector<HP> b){int n=b.size();for(int i=0;i<n;i++){int p=i;HP best=abs(a[i][i]);for(int j=i+1;j<n;j++)if(abs(a[j][i])>best){p=j;best=abs(a[j][i]);}if(best<HP("1e-52"))throw runtime_error("Rank deficient augmented matrix");swap(a[i],a[p]);swap(b[i],b[p]);HP pivot=a[i][i];for(int j=i;j<n;j++)a[i][j]/=pivot;b[i]/=pivot;for(int k=i+1;k<n;k++){HP t=a[k][i];if(t==0)continue;for(int j=i+1;j<n;j++)a[k][j]-=t*a[i][j];a[k][i]=0;b[k]-=t*b[i];}}vector<HP>x(n);for(int i=n-1;i>=0;i--){HP t=b[i];for(int j=i+1;j<n;j++)t-=a[i][j]*x[j];x[i]=t;}return x;}
vector<double> weights(const vector<Offset>& s,int q,int m,bool logform=false){auto ps=powers(m);int ns=s.size(),np=ps.size(),n=ns+np;if(ns<np)throw runtime_error("Too few stencil nodes for polynomial space");vector<vector<HP>> a(n,vector<HP>(n));vector<HP>b(n);for(int i=0;i<ns;i++){for(int j=0;j<ns;j++){int dx=s[i].a-s[j].a,dy=s[i].b-s[j].b;a[i][j]=kernel(HP(dx*dx+dy*dy),q,logform);}b[i]=lapkernel(HP(s[i].a*s[i].a+s[i].b*s[i].b),q,logform);for(int j=0;j<np;j++){auto [px,py]=ps[j];HP v=ipow(HP(s[i].a),px)*ipow(HP(s[i].b),py);a[i][ns+j]=v;a[ns+j][i]=v;}}for(int j=0;j<np;j++){auto [px,py]=ps[j];b[ns+j]=(px==2&&py==0)||(px==0&&py==2)?2:0;}auto x=solve(a,b);vector<double>w(ns);for(int i=0;i<ns;i++)w[i]=x[i].convert_to<double>();return w;}
struct Method{string name;vector<Offset>s;vector<double>w;int q,m;bool logform;};
Method make(string name,int r2,int q,int m,bool lg=false){Method z{name,shell(r2),{},q,m,lg};z.w=weights(z.s,q,m,lg);return z;}
double frequency(const Method& m,int k,int l,int N){double h=PI/(N-1),sum=0;for(size_t j=0;j<m.s.size();j++)sum+=m.w[j]*cos(k*m.s[j].a*h)*cos(l*m.s[j].b*h);return -sum/(h*h);}
double fd4frequency(int k,int l,int N){double h=PI/(N-1),t=pow(sin(k*h/2),2),s=pow(sin(l*h/2),2);return (4*(t+s)+(4.0/3)*(t*t+s*s))/(h*h);}
using Mode=tuple<int,int,double>;
vector<Mode> mixed(){return {{1,1,1.0},{2,3,.8},{1,5,.5},{3,4,.4},{5,5,.2}};}
vector<Mode> favorable(){return {{2,3,.8},{3,4,.4}};}
complex<double> rk4(complex<double> z,int n){complex<double> r=1.0+z+z*z/2.0+z*z*z/6.0+z*z*z*z/24.0,v=1.;for(int i=0;i<n;i++)v*=r;return v;}
struct Result{double error,temporal;};
Result modeerror(const Method* m,int N,double T,const vector<Mode>& modes,bool fd=false){double h=PI/(N-1),omegaMax=0;for(auto [k,l,c]:modes)omegaMax=max(omegaMax,fd?fd4frequency(k,l,N):frequency(*m,k,l,N));double dt=.0007;
// RK4 imaginary-axis stability; use maximum symbol sampled on all discrete sine frequencies.
double spectralMax=0;if(fd){spectralMax=(4*2+4.0/3*2)/(h*h);}else{for(int k=1;k<N-1;k++)for(int l=1;l<N-1;l++)spectralMax=max(spectralMax,abs(frequency(*m,k,l,N)));}dt=min(.0007,2.4/spectralMax);int steps=max(1,(int)ceil(T/dt));dt=T/steps;
vector<complex<double>> numer,semi,exact;for(auto [k,l,c]:modes){double wh=fd?fd4frequency(k,l,N):frequency(*m,k,l,N);numer.push_back(c*rk4(complex<double>(0,-wh*dt),steps));semi.push_back(c*exp(complex<double>(0,-wh*T)));exact.push_back(c*exp(complex<double>(0,-(k*k+l*l)*T)));}
double err=0,den=0,terr=0;for(int i=1;i<N-1;i++)for(int j=1;j<N-1;j++){complex<double> a=0,b=0,c=0;for(size_t z=0;z<modes.size();z++){auto [k,l,amp]=modes[z];double s=sin(k*i*h)*sin(l*j*h);a+=numer[z]*s;b+=exact[z]*s;c+=semi[z]*s;}err+=norm(a-b);den+=norm(b);terr+=norm(a-c);}return {sqrt(err/den),sqrt(terr/den)};}
void csvheader(ofstream& f,const string& h){f<<h<<'\n'<<setprecision(16);}
struct Matrix{int n;vector<map<int,double>> rows;};
int reflect(int i,int n,int& sign){while(i<0||i>n){if(i<0){i=-i;sign=-sign;}if(i>n){i=2*n-i;sign=-sign;}}return i;}
Matrix reflected(const Method& m,int N){int n=N-2;double h=PI/(N-1);Matrix L{n*n,vector<map<int,double>>(n*n)};for(int i=1;i<=n;i++)for(int j=1;j<=n;j++){int row=(i-1)*n+j-1;for(size_t z=0;z<m.s.size();z++){int sign=1,ii=reflect(i+m.s[z].a,N-1,sign),jj=reflect(j+m.s[z].b,N-1,sign);if(ii==0||jj==0||ii==N-1||jj==N-1)continue;int col=(ii-1)*n+jj-1;L.rows[row][col]+=sign*m.w[z]/(h*h);}}return L;}
// Rank-safe KNN: first select a polynomially unisolvent subset, then fill to 29.
// Orthogonalization is only for geometric rank testing; actual weights use 65-digit solves.
bool append_independent(const Offset& o, const vector<pair<int,int>>& ps, vector<vector<double>>& Q){
 vector<double> v;v.reserve(ps.size());for(auto [a,b]:ps)v.push_back(pow(o.a/5.0,a)*pow(o.b/5.0,b));
 double orig=inner_product(v.begin(),v.end(),v.begin(),0.0);
 for(int pass=0;pass<2;pass++)for(auto &q:Q){double d=inner_product(v.begin(),v.end(),q.begin(),0.0);for(size_t k=0;k<v.size();k++)v[k]-=d*q[k];}
 double n2=inner_product(v.begin(),v.end(),v.begin(),0.0);
 if(n2<1e-20*max(1.0,orig))return false;
 for(double &x:v)x/=sqrt(n2);Q.push_back(move(v));return true;
}
vector<Offset> knnstencil(int i,int j,int N,int ns,int poly){
 vector<tuple<int,int,int>> a;for(int x=0;x<N;x++)for(int y=0;y<N;y++)a.push_back({(x-i)*(x-i)+(y-j)*(y-j),x,y});sort(a.begin(),a.end());
 auto ps=powers(poly);vector<vector<double>> Q;vector<Offset>s;set<pair<int,int>> chosen;
 // Require polynomial rank before adding optional nearest neighbors.
 for(auto [d,x,y]:a){Offset o{x-i,y-j};if(append_independent(o,ps,Q)){s.push_back(o);chosen.insert({x,y});if(Q.size()==ps.size())break;}}
 if(Q.size()!=ps.size())throw runtime_error("Cannot find polynomially unisolvent KNN stencil");
 for(auto [d,x,y]:a){if((int)s.size()==ns)break;if(!chosen.count({x,y})){s.push_back({x-i,y-j});chosen.insert({x,y});}}
 if((int)s.size()!=ns)throw runtime_error("Too few nodes for KNN stencil");return s;
}
Matrix knn(int N,int ns,int q,int m){int n=N-2;double h=PI/(N-1);Matrix L{n*n,vector<map<int,double>>(n*n)};for(int i=1;i<=n;i++)for(int j=1;j<=n;j++){auto s=knnstencil(i,j,N,ns,m);auto w=weights(s,q,m);int row=(i-1)*n+j-1;for(size_t z=0;z<s.size();z++){int x=i+s[z].a,y=j+s[z].b;if(x>0&&x<N-1&&y>0&&y<N-1)L.rows[row][(x-1)*n+y-1]+=w[z]/(h*h);}}return L;}
void savematrix(const Matrix& L,const string& path){ofstream f(path);csvheader(f,"row,col,value");for(int i=0;i<L.n;i++)for(auto [j,v]:L.rows[i])if(abs(v)>1e-14)f<<i<<','<<j<<','<<v<<'\n';}
vector<double> matvec(const Matrix& L,const vector<double>& u){vector<double>v(L.n);for(int i=0;i<L.n;i++)for(auto [j,w]:L.rows[i])v[i]+=w*u[j];return v;}
vector<double> sample(int N,int k,int l){int n=N-2;double h=PI/(N-1);vector<double>v(n*n);for(int i=1;i<=n;i++)for(int j=1;j<=n;j++)v[(i-1)*n+j-1]=sin(k*i*h)*sin(l*j*h);return v;}
void diagnostics(const Matrix& L,int N,const string& name,const string& dir){auto phi=sample(N,2,3),v=matvec(L,phi);double dot=0,normphi=0,res=0,shape=0,layer=0,all=0;for(int i=0;i<L.n;i++){dot+=phi[i]*v[i];normphi+=phi[i]*phi[i];}double wh=-dot/normphi;int n=N-2;ofstream f(dir+"/boundary_"+name+".csv");csvheader(f,"layer,max_residual");vector<double>mx((n+1)/2+1);for(int i=0;i<n;i++)for(int j=0;j<n;j++){int z=i*n+j,layer=min({i+1,j+1,n-i,n-j});double r=abs(v[z]+13*phi[z]);mx[layer]=max(mx[layer],r);res=max(res,r);shape+=pow(v[z]+wh*phi[z],2);}for(size_t i=1;i<mx.size();i++)f<<i<<','<<mx[i]<<'\n';double sy=0,allnorm=0;for(int i=0;i<L.n;i++){for(auto [j,w]:L.rows[i]){allnorm+=w*w;auto it=L.rows[j].find(i);double other=it==L.rows[j].end()?0:it->second;sy+=pow(w-other,2);}}ofstream g(dir+"/diagnostics_"+name+".csv");csvheader(g,"N,symmetry_defect,mode_frequency,mode_frequency_error,shape_residual_l2,mode_residual_inf");g<<N<<','<<sqrt(sy/allnorm)<<','<<wh<<','<<abs(wh-13)<<','<<sqrt(shape)<<','<<res<<'\n';}
double dotv(const vector<double>&a,const vector<double>&b){return inner_product(a.begin(),a.end(),b.begin(),0.0);}
vector<double> axpy(const vector<double>&a,const vector<double>&b,double alpha){vector<double>z(a.size());for(size_t i=0;i<a.size();i++)z[i]=a[i]+alpha*b[i];return z;}
// BiCGSTAB for the general nonsymmetric KNN matrix. No symmetry is assumed.
vector<double> bicgstab(const Matrix&L,const vector<double>&b,double alpha,const vector<double>&guess){
 auto apply=[&](const vector<double>&x){auto y=matvec(L,x);auto z=matvec(L,y);for(size_t i=0;i<x.size();i++)z[i]=x[i]+alpha*z[i];return z;};
 vector<double>x=guess,r=axpy(b,apply(x),-1),r0=r,p(r.size()),v(r.size());
 double normb=sqrt(dotv(b,b)),rho=1,omega=1,aa=1;
 if(sqrt(dotv(r,r))<=1e-11*max(1.0,normb))return x;
 for(int it=0;it<250;it++){
  double rhonew=dotv(r0,r);if(abs(rhonew)<1e-28)break;
  double beta=(rhonew/rho)*(aa/omega);
  for(size_t i=0;i<p.size();i++)p[i]=r[i]+beta*(p[i]-omega*v[i]);
  v=apply(p);double den=dotv(r0,v);if(abs(den)<1e-28)break;aa=rhonew/den;
  auto s=axpy(r,v,-aa);
  if(sqrt(dotv(s,s))<=1e-11*max(1.0,normb)){for(size_t i=0;i<x.size();i++)x[i]+=aa*p[i];return x;}
  auto t=apply(s);double tt=dotv(t,t);if(tt<1e-28)break;omega=dotv(t,s)/tt;
  for(size_t i=0;i<x.size();i++)x[i]+=aa*p[i]+omega*s[i];
  r=axpy(s,t,-omega);if(sqrt(dotv(r,r))<=1e-11*max(1.0,normb))return x;
  if(abs(omega)<1e-28)break;rho=rhonew;
 }
 throw runtime_error("Midpoint BiCGSTAB failed to converge");
}
void trajectory(const Matrix& L,int N,const string& name,const string& dir,double T,double dt,string integrator){
 int n=L.n;auto u0=sample(N,2,3);vector<double>y(2*n);copy(u0.begin(),u0.end(),y.begin());
 auto rhs=[&](const vector<double>& z){vector<double>a(z.begin(),z.begin()+n),b(z.begin()+n,z.end());auto x=matvec(L,b),v=matvec(L,a);vector<double>r(2*n);for(int i=0;i<n;i++){r[i]=-x[i];r[n+i]=v[i];}return r;};
 auto mass=[&](const vector<double>&z){return dotv(z,z);};
 auto energy=[&](const vector<double>&z){vector<double>u(z.begin(),z.begin()+n),v(z.begin()+n,z.end());auto Lu=matvec(L,u),Lv=matvec(L,v);return -(dotv(u,Lu)+dotv(v,Lv));};
 double mass0=mass(y),energy0=energy(y);int steps=(int)ceil(T/dt);dt=T/steps;
 ofstream f(dir+"/trajectory_"+name+"_"+integrator+"_T"+to_string((int)T)+".csv");csvheader(f,"time,combined_rel_l2_error,mass_relative_drift,energy_relative_drift");
 for(int it=0;it<=steps;it++){
  if(it%max(1,steps/200)==0||it==steps){double t=it*dt,e=0,den=0;for(int i=0;i<n;i++){double eu=cos(13*t)*u0[i],ev=-sin(13*t)*u0[i];e+=pow(y[i]-eu,2)+pow(y[n+i]-ev,2);den+=eu*eu+ev*ev;}f<<t<<','<<sqrt(e/den)<<','<<abs(mass(y)-mass0)/mass0<<','<<abs(energy(y)-energy0)/max(1e-30,abs(energy0))<<'\n';}
  if(it==steps)break;
  if(integrator=="RK4"){
   auto k1=rhs(y);vector<double>tmp(2*n);for(int i=0;i<2*n;i++)tmp[i]=y[i]+dt*k1[i]/2;auto k2=rhs(tmp);for(int i=0;i<2*n;i++)tmp[i]=y[i]+dt*k2[i]/2;auto k3=rhs(tmp);for(int i=0;i<2*n;i++)tmp[i]=y[i]+dt*k3[i];auto k4=rhs(tmp);for(int i=0;i<2*n;i++)y[i]+=dt*(k1[i]+2*k2[i]+2*k3[i]+k4[i])/6;
  }else{
   // Cayley midpoint: (I+dt² L²/4) u_{n+1}=(I-dt² L²/4)u_n-dt L v_n;
   // then v_{n+1}=v_n+dt L (u_n+u_{n+1})/2.
   vector<double>u(y.begin(),y.begin()+n),v(y.begin()+n,y.end());auto Lu=matvec(L,u),L2u=matvec(L,Lu),Lv=matvec(L,v);
   vector<double>b(n);for(int i=0;i<n;i++)b[i]=u[i]-dt*dt*L2u[i]/4-dt*Lv[i];
   auto unew=bicgstab(L,b,dt*dt/4,u);auto LuNew=matvec(L,unew);
   for(int i=0;i<n;i++){y[i]=unew[i];y[n+i]=v[i]+dt*(Lu[i]+LuNew[i])/2;}
  }
 }
}
int main(int argc,char**argv){try{
 string out=argc>1?argv[1]:"data";filesystem::create_directories(out);
 auto r7=make("r7_P4",9,7,4),r11=make("r11_P5",9,11,5);
 ofstream screen(out+"/basis_screening.csv");csvheader(screen,"kernel,poly_degree,stencil_points,N,T,relative_l2_error,temporal_error");
 for(auto [q,m,lg]:vector<tuple<int,int,bool>>{{3,2,0},{5,3,0},{7,4,0},{9,4,0},{11,5,0},{2,1,1},{4,2,1},{6,3,1},{8,4,1},{10,5,1}}){auto z=make("test",9,q,m,lg);auto r=modeerror(&z,31,1,mixed());screen<<(lg?"r"+to_string(q)+"logr":"r"+to_string(q))<<','<<m<<','<<z.s.size()<<",31,1,"<<r.error<<','<<r.temporal<<'\n';}screen.close();
 ofstream f(out+"/modal_rk4.csv");csvheader(f,"case,N,T,method,relative_l2_error,temporal_error");
 for(auto scenario:vector<pair<string,vector<Mode>>>{{"mixed",mixed()},{"favorable",favorable()}})for(int N:{21,31,41,61,81,101})for(double T:{1.,20.})for(auto method:vector<pair<string,const Method*>>{{"r11_P5",&r11},{"r7_P4",&r7},{"FD4",nullptr}}){auto r=modeerror(method.second,N,T,scenario.second,method.second==nullptr);f<<scenario.first<<','<<N<<','<<T<<','<<method.first<<','<<r.error<<','<<r.temporal<<'\n';}f.close();
 ofstream w(out+"/weights_29.csv");csvheader(w,"kernel,a,b,weight");for(auto m:{r7,r11})for(size_t i=0;i<m.s.size();i++)w<<m.name<<','<<m.s[i].a<<','<<m.s[i].b<<','<<m.w[i]<<'\n';
 ofstream mom(out+"/moments.csv");csvheader(mom,"kernel,M40,M22,M60,M42,frequency_k2_l3_N31");for(auto m:{r7,r11}){double M40=0,M22=0,M60=0,M42=0;for(size_t i=0;i<m.s.size();i++){auto o=m.s[i];double z=m.w[i];M40+=z*pow(o.a,4);M22+=z*pow(o.a,2)*pow(o.b,2);M60+=z*pow(o.a,6);M42+=z*pow(o.a,4)*pow(o.b,2);}mom<<m.name<<','<<M40<<','<<M22<<','<<M60<<','<<M42<<','<<frequency(m,2,3,31)<<'\n';}mom.close();
 // Main boundary-treatment comparison uses r11+P5 exclusively, including KNN.
 // N=21,31 give direct sparse matrix comparison; higher grids use modal method.
 for(int N:{21,31}){
  cerr<<"Assembling r11+P5 reflected N="<<N<<"\n";
  auto R=reflected(r11,N);string rn="Reflected_r11_P5_N"+to_string(N);
  savematrix(R,out+"/matrix_"+rn+".csv");diagnostics(R,N,rn,out);
  cerr<<"Assembling rank-safe r11+P5 KNN N="<<N<<"\n";
  auto K=knn(N,29,11,5);string kn="KNN_r11_P5_N"+to_string(N);
  savematrix(K,out+"/matrix_"+kn+".csv");diagnostics(K,N,kn,out);
  for(auto item:vector<pair<string,const Matrix*>>{{kn,&K},{rn,&R}}){
   trajectory(*item.second,N,item.first,out,2.0,.0005,"RK4");
   trajectory(*item.second,N,item.first,out,2.0,.005,"Midpoint");
   if(N==21)trajectory(*item.second,N,item.first,out,20.0,.0005,"RK4");
  }
 }
 cerr<<"Completed CONSISTENT r11+P5 C++ experiments: "<<out<<'\n';
 }catch(const exception&e){cerr<<"ERROR: "<<e.what()<<'\n';return 1;}}
