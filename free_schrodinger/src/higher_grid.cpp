#define main legacy_main
#include "main.cpp"
#undef main
#include <chrono>
#include <sstream>
#include <unordered_map>

// A separate executable preserves the original v3 experiments unchanged.
// Tie-break variants change only ordering among equally distant candidates.
vector<Offset> selected_stencil(int i,int j,int N,int ns,int poly,int tie_mode){
 vector<tuple<int,int,int>> a;
 for(int x=max(0,i-10);x<=min(N-1,i+10);x++)for(int y=max(0,j-10);y<=min(N-1,j+10);y++)if((x-i)*(x-i)+(y-j)*(y-j)<=100)a.push_back({(x-i)*(x-i)+(y-j)*(y-j),x,y});
 sort(a.begin(),a.end(),[&](auto A,auto B){
  if(get<0>(A)!=get<0>(B))return get<0>(A)<get<0>(B);
  int x=get<1>(A),y=get<2>(A),xx=get<1>(B),yy=get<2>(B);
  if(tie_mode==0)return std::tie(x,y)<std::tie(xx,yy);
  if(tie_mode==1)return std::tie(y,x)<std::tie(yy,xx);
  return std::tie(x,y)>std::tie(xx,yy);
 });
 auto ps=powers(poly);vector<vector<double>> Q;vector<Offset>s;set<pair<int,int>> chosen;
 for(auto [d,x,y]:a){Offset o{x-i,y-j};if(append_independent(o,ps,Q)){s.push_back(o);chosen.insert({x,y});if(Q.size()==ps.size())break;}}
 if(Q.size()!=ps.size())throw runtime_error("Rank failure");
 for(auto [d,x,y]:a){if((int)s.size()==ns)break;if(!chosen.count({x,y})){s.push_back({x-i,y-j});chosen.insert({x,y});}}
 if((int)s.size()!=ns)throw runtime_error("Not enough nodes");return s;
}
string key_for(const vector<Offset>&s){string k;for(auto o:s){k+=to_string(o.a)+":"+to_string(o.b)+";";}return k;}
struct Stats {Matrix L;size_t unique;double assembly_s;};
Stats cached_knn(int N,int ns,int tie){
 int n=N-2;double h=PI/(N-1);Matrix L{n*n,vector<map<int,double>>(n*n)};
 unordered_map<string,vector<double>> cache;auto start=chrono::steady_clock::now();
 for(int i=1;i<=n;i++)for(int j=1;j<=n;j++){
  auto s=selected_stencil(i,j,N,ns,5,tie);auto key=key_for(s);auto it=cache.find(key);
  if(it==cache.end())it=cache.emplace(key,weights(s,11,5)).first;
  auto &w=it->second;int row=(i-1)*n+j-1;
  for(size_t z=0;z<s.size();z++){int x=i+s[z].a,y=j+s[z].b;if(x>0&&x<N-1&&y>0&&y<N-1)L.rows[row][(x-1)*n+y-1]+=w[z]/(h*h);}
 }
 return {move(L),cache.size(),chrono::duration<double>(chrono::steady_clock::now()-start).count()};
}
struct Measures{double relsym,rayleigh,shape,layer1,layer2,global;};
Measures measure(const Matrix&L,int N){
 auto phi=sample(N,2,3),v=matvec(L,phi);double pp=dotv(phi,phi),wh=-dotv(phi,v)/pp,shape=0,global=0,l1=0,l2=0;
 int n=N-2;for(int i=0;i<n;i++)for(int j=0;j<n;j++){int z=i*n+j,layer=min({i+1,j+1,n-i,n-j});double r=abs(v[z]+13*phi[z]);global=max(global,r);if(layer==1)l1=max(l1,r);if(layer==2)l2=max(l2,r);shape+=pow(v[z]+wh*phi[z],2);}
 double sy=0,all=0;for(int i=0;i<L.n;i++)for(auto [j,w]:L.rows[i]){all+=w*w;auto it=L.rows[j].find(i);double t=it==L.rows[j].end()?0:it->second;sy+=pow(w-t,2);}
 return {sqrt(sy/all),abs(wh-13),sqrt(shape/pp),l1,l2,global};
}
// Time integration of a single sine mode, using a timestep selected from h^2.
// This is a sparse full-state RK4 experiment, not the DST modal surrogate.
double rk4_final(const Matrix&L,int N,double T,double dt){
 int n=L.n;auto phi=sample(N,2,3);vector<double>u=phi,v(n),k1u(n),k1v(n),k2u(n),k2v(n),k3u(n),k3v(n),k4u(n),k4v(n),a(n),b(n);
 int steps=(int)ceil(T/dt);dt=T/steps;
 for(int it=0;it<steps;it++){
  auto Lv=matvec(L,v),Lu=matvec(L,u);
  for(int i=0;i<n;i++){k1u[i]=-Lv[i];k1v[i]=Lu[i];a[i]=u[i]+dt*k1u[i]/2;b[i]=v[i]+dt*k1v[i]/2;}
  Lv=matvec(L,b);Lu=matvec(L,a);
  for(int i=0;i<n;i++){k2u[i]=-Lv[i];k2v[i]=Lu[i];a[i]=u[i]+dt*k2u[i]/2;b[i]=v[i]+dt*k2v[i]/2;}
  Lv=matvec(L,b);Lu=matvec(L,a);
  for(int i=0;i<n;i++){k3u[i]=-Lv[i];k3v[i]=Lu[i];a[i]=u[i]+dt*k3u[i];b[i]=v[i]+dt*k3v[i];}
  Lv=matvec(L,b);Lu=matvec(L,a);
  for(int i=0;i<n;i++){k4u[i]=-Lv[i];k4v[i]=Lu[i];u[i]+=dt*(k1u[i]+2*k2u[i]+2*k3u[i]+k4u[i])/6;v[i]+=dt*(k1v[i]+2*k2v[i]+2*k3v[i]+k4v[i])/6;}
 }
 double e=0,p=dotv(phi,phi);for(int i=0;i<n;i++)e+=pow(u[i]-cos(13*T)*phi[i],2)+pow(v[i]+sin(13*T)*phi[i],2);
 return sqrt(e/p);
}
int main(int argc,char**argv){try{
 string out=argc>1?argv[1]:"data";filesystem::create_directories(out);
 int maxN=argc>2?stoi(argv[2]):61; int minN=argc>3?stoi(argv[3]):21;
 ofstream f(out+"/higher_grid_knn_sensitivity.csv");csvheader(f,"N,method,stencil_points,tie_break,unique_stencils,assembly_seconds,symmetry_defect,frequency_error_k2_l3,shape_residual_rel_l2,boundary_layer1_inf,boundary_layer2_inf,global_residual_inf,rk4_T1_dt,rk4_T1_rel_l2_error");
 auto ref=make("r11_P5",9,11,5);
 for(int N:{21,31,41,61,81,101}){
  if(N>maxN||N<minN)continue;
  cerr<<"Higher-grid N="<<N<<"\n";
  auto t0=chrono::steady_clock::now();auto R=reflected(ref,N);double sec=chrono::duration<double>(chrono::steady_clock::now()-t0).count();
  auto mr=measure(R,N);double dt=min(.0005,.12*pow(PI/(N-1),2));
  double err= N<=41?rk4_final(R,N,1.,dt):numeric_limits<double>::quiet_NaN();
  f<<N<<",Reflected,29,complete_shell,1,"<<sec<<','<<mr.relsym<<','<<mr.rayleigh<<','<<mr.shape<<','<<mr.layer1<<','<<mr.layer2<<','<<mr.global<<','<<dt<<','<<err<<'\n';f.flush();
  for(int ns:{29,35,41})for(int tie=0;tie<3;tie++){
   cerr<<"  KNN n="<<ns<<" tie="<<tie<<"\n";
   auto S=cached_knn(N,ns,tie);auto m=measure(S.L,N);
   // Full sparse RK4 on higher grids is intentionally restricted to baseline 29-point KNN.
   double error=(ns==29&&tie==0&&N<=41)?rk4_final(S.L,N,1.,dt):numeric_limits<double>::quiet_NaN();
   f<<N<<",KNN,"<<ns<<','<<(tie==0?"xy_ascending":tie==1?"yx_ascending":"xy_descending")<<','<<S.unique<<','<<S.assembly_s<<','<<m.relsym<<','<<m.rayleigh<<','<<m.shape<<','<<m.layer1<<','<<m.layer2<<','<<m.global<<','<<dt<<','<<error<<'\n';f.flush();
  }
 }
 cerr<<"Completed higher-grid and KNN sensitivity study: "<<out<<"\n";
 }catch(const exception&e){cerr<<"ERROR: "<<e.what()<<'\n';return 1;}}
