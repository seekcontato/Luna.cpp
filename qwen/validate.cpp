// Validador GGUF F16 do Qwen3.5-0.8B. Nao executa inferencia.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
using namespace std;
static void require(bool ok,const string& err){if(!ok) throw runtime_error(err);}
struct Reader {
 ifstream f; uint64_t size=0,pos=0; static constexpr uint64_t max_header=64ull*1024*1024;
 explicit Reader(const char*p):f(p,ios::binary){require(bool(f),"nao abriu arquivo");f.seekg(0,ios::end);auto n=f.tellg();require(n>=0,"nao mediu arquivo");size=uint64_t(n);f.seekg(0);}
 void read(char*p,uint64_t n){require(pos<=size && n<=size-pos && pos<=max_header && n<=max_header-pos,"cabecalho truncado ou grande");if(n)require(bool(f.read(p,streamsize(n))),"falha leitura");pos+=n;}
 uint64_t u(int n){unsigned char b[8]={};read((char*)b,n);uint64_t v=0;for(int i=0;i<n;i++)v|=uint64_t(b[i])<<(8*i);return v;}
 string str(){auto n=u(8);require(n<=65536,"string grande");string s(size_t(n),'\0');read(s.data(),n);require(s.find('\0')==string::npos,"string com NUL");return s;}
};
struct Val { uint32_t type=0,elem=0; uint64_t u=0; double num=0; string s; vector<Val> a; };
static Val value(Reader&r,uint32_t t,bool nested=false){
 Val v;v.type=t;
 switch(t){
 case 0:case 1:case 7:v.u=r.u(1);if(t==7)require(v.u<=1,"bool invalido");break;
 case 2:case 3:v.u=r.u(2);break;
 case 4:case 5:v.u=r.u(4);break;
 case 6:{uint32_t bits=uint32_t(r.u(4));float x;memcpy(&x,&bits,4);v.num=x;break;}
 case 8:v.s=r.str();break;
 case 9:{require(!nested,"array aninhado");v.elem=uint32_t(r.u(4));auto n=r.u(8);require(n<=200000 && v.elem<=12 && v.elem!=9,"array invalido");v.a.reserve(size_t(n));for(uint64_t i=0;i<n;i++)v.a.push_back(value(r,v.elem,true));break;}
 case 10:case 11:v.u=r.u(8);break;
 case 12:{uint64_t bits=r.u(8);double x;memcpy(&x,&bits,8);v.num=x;break;}
 default:throw runtime_error("tipo de metadado desconhecido");
 }return v;
}
struct Tensor {string name; vector<uint64_t> dims;uint32_t type;uint64_t offset,bytes;};
#ifdef LUNA_VISUAL_CAPTURE
static vector<Tensor> captured_tensors;
static uint64_t captured_data = 0;
#endif
static map<string,Tensor> schema(){
 map<string,Tensor> m;
 auto add=[&](string n,uint32_t t,vector<uint64_t>d){m.emplace(n,Tensor{n,d,t,0,0});};
 for(int i=0;i<12;i++){
 string p="v.blk."+to_string(i)+".";
 for(auto n:{"ln1","ln2"}){add(p+n+".weight",0,{768});add(p+n+".bias",0,{768});}
 for(auto item:vector<pair<string,vector<uint64_t>>>{{"attn_out",{768,768}},{"attn_qkv",{768,2304}},{"ffn_up",{768,3072}},{"ffn_down",{3072,768}}}){
 add(p+item.first+".weight",1,item.second);add(p+item.first+".bias",0,{item.second[1]});}
 }
 add("mm.0.weight",1,{3072,3072});add("mm.0.bias",0,{3072});
 add("mm.2.weight",1,{3072,1024});add("mm.2.bias",0,{1024});
 add("v.post_ln.weight",0,{768});add("v.post_ln.bias",0,{768});
 add("v.patch_embd.bias",0,{768});add("v.patch_embd.weight",1,{16,16,3,768});add("v.patch_embd.weight.1",1,{16,16,3,768});
 add("v.position_embd.weight",0,{768,2304});return m;
}
// Entrada: um patch ja normalizado, F32 little-endian [3,2,16,16], sem cabecalho.
// O resultado e apenas a primeira Conv3D, nao um embedding visual final.
static void patch_embed(Reader& r, uint64_t data, const vector<Tensor>& ts, const char* input) {
 auto get=[&](const string& n)->const Tensor&{for(const auto&t:ts)if(t.name==n)return t;throw runtime_error("tensor de patch ausente");};
 auto load=[&](const Tensor&t){vector<unsigned char>b(size_t(t.bytes));r.f.clear();r.f.seekg(streamoff(data+t.offset));require(bool(r.f) && bool(r.f.read((char*)b.data(),streamsize(b.size()))),"leitura de pesos falhou");return b;};
 auto f32=[](const unsigned char*p){uint32_t b=uint32_t(p[0])|(uint32_t(p[1])<<8)|(uint32_t(p[2])<<16)|(uint32_t(p[3])<<24);float x;memcpy(&x,&b,4);return x;};
 auto f16=[](const unsigned char*p){unsigned h=unsigned(p[0])|(unsigned(p[1])<<8);unsigned e=(h>>10)&31,m=h&1023;require(e!=31,"peso F16 NaN/Inf");double x=e?ldexp(1.0+double(m)/1024.0,int(e)-15):ldexp(double(m),-24);return (h&0x8000)?-x:x;};
 auto a=load(get("v.patch_embd.weight"));auto b=load(get("v.patch_embd.weight.1"));auto bias=load(get("v.patch_embd.bias"));
 ifstream in(input,ios::binary);require(bool(in),"patch nao abriu");vector<unsigned char> raw(1536*4);require(bool(in.read((char*)raw.data(),raw.size())),"patch truncado");require(in.peek()==char_traits<char>::eof() && !in.bad(),"patch deve ter exatamente 6144 bytes");
 vector<float>x(1536);for(size_t i=0;i<x.size();i++){x[i]=f32(&raw[i*4]);require(isfinite(x[i]) && x[i]>=-1 && x[i]<=1,"patch deve conter F32 finito em [-1,1]");}
 vector<double> output(768);
 for(size_t o=0;o<768;o++){
  double sum=f32(&bias[o*4]);require(isfinite(sum),"bias NaN/Inf");
  for(size_t c=0;c<3;c++)for(size_t k=0;k<256;k++){
   size_t wi=(o*768+c*256+k)*2, xi=c*512+k;
   sum+=f16(&a[wi])*x[xi]+f16(&b[wi])*x[xi+256];
  }
  require(isfinite(sum),"saida de patch nao finita");output[o]=sum;
 }
 cout.precision(17);for(size_t o=0;o<768;o++)cout<<"PATCH "<<o<<" "<<output[o]<<"\n";
}
static void validate(const char*path,bool scan,const char*patch){
 Reader r(path);char m[4];r.read(m,4);require(memcmp(m,"GGUF",4)==0,"magic GGUF invalido");require(r.u(4)==3,"apenas GGUF v3 little-endian");auto nt=r.u(8),nm=r.u(8);require(nt==154 && nm<=10000,"numero de tensores ou metadados invalido");
 map<string,Val> kv;
 for(uint64_t i=0;i<nm;i++){auto k=r.str();auto t=uint32_t(r.u(4));Val v=value(r,t);require(kv.emplace(k,move(v)).second,"chave duplicada: "+k);}
 auto get=[&](string k,uint32_t t)->const Val&{auto i=kv.find(k);require(i!=kv.end(),"metadado ausente: "+k);require(i->second.type==t,"tipo de metadado invalido: "+k);return i->second;};
 auto s=[&](string k,string x){require(get(k,8).s==x,"valor incompatível: "+k);};
 auto u=[&](string k,uint32_t t,uint64_t x){require(get(k,t).u==x,"valor incompatível: "+k);};
 s("general.architecture","clip");s("general.type","mmproj");s("clip.projector_type","qwen3vl_merger");
 u("general.file_type",4,1);u("clip.has_vision_encoder",7,1);u("clip.use_gelu",7,1);
 for(auto p:vector<pair<string,uint64_t>>{{"clip.vision.projection_dim",1024},{"clip.vision.patch_size",16},{"clip.vision.embedding_length",768},{"clip.vision.feed_forward_length",3072},{"clip.vision.block_count",12},{"clip.vision.attention.head_count",12},{"clip.vision.spatial_merge_size",2}})u(p.first,4,p.second);
 require(fabs(get("clip.vision.attention.layer_norm_epsilon",6).num-1e-6)<1e-11,"epsilon diferente");
 for(string k:{"clip.vision.image_mean","clip.vision.image_std"}){auto&v=get(k,9);require(v.elem==6 && v.a.size()==3,"normalizacao invalida");for(auto&x:v.a)require(x.num==0.5,"normalizacao diferente");}
 auto& deep=get("clip.vision.is_deepstack_layers",9);require(deep.elem==7 && deep.a.size()==12,"deepstack invalido");for(auto&x:deep.a)require(x.u==0,"deepstack nao suportado");
 uint64_t align=32;auto a=kv.find("general.alignment");if(a!=kv.end()){require(a->second.type==4,"tipo de alinhamento");align=a->second.u;}require(align && align<=1048576 && !(align&(align-1)),"alinhamento invalido");
 auto expected=schema();vector<Tensor> tensors;set<string> names;
 for(uint64_t i=0;i<nt;i++){
 Tensor t;t.name=r.str();require(names.insert(t.name).second,"tensor duplicado: "+t.name);auto nd=r.u(4);require(nd>=1&&nd<=4,"rank invalido");uint64_t elements=1;
 for(uint64_t j=0;j<nd;j++){auto d=r.u(8);require(d && d<=UINT64_MAX/elements,"dimensao invalida");elements*=d;t.dims.push_back(d);}
 t.type=uint32_t(r.u(4));t.offset=r.u(8);auto e=expected.find(t.name);require(e!=expected.end(),"tensor inesperado: "+t.name);require(t.type==e->second.type && t.dims==e->second.dims,"tipo/shape divergente: "+t.name);uint64_t b=t.type==0?4:2;require(elements<=UINT64_MAX/b,"overflow");t.bytes=elements*b;expected.erase(e);tensors.push_back(t);
 }
 require(expected.empty(),"tensor ausente");uint64_t data=(r.pos+align-1)&~(align-1);require(data<=r.size,"dados ausentes");
 sort(tensors.begin(),tensors.end(),[](const Tensor&a,const Tensor&b){return a.offset<b.offset;});uint64_t end=0;
 for(auto&t:tensors){require(t.offset%align==0 && t.offset>=end,"offset desalinhado ou sobreposto: "+t.name);require(t.offset<=r.size-data && t.bytes<=r.size-data-t.offset,"tensor truncado: "+t.name);end=t.offset+t.bytes;}
 if(scan){
  vector<unsigned char> buf(65536);
  for(auto&t:tensors){r.f.clear();r.f.seekg(streamoff(data+t.offset));require(bool(r.f),"seek falhou");uint64_t left=t.bytes;
   while(left){size_t n=size_t(min<uint64_t>(left,buf.size()));require(bool(r.f.read((char*)buf.data(),n)),"falha leitura dados");
    if(t.type==1){for(size_t i=0;i<n;i+=2){unsigned x=unsigned(buf[i])|(unsigned(buf[i+1])<<8);require((x&0x7c00)!=0x7c00,"NaN/Inf: "+t.name);}}
    else {for(size_t i=0;i<n;i+=4){uint32_t x=uint32_t(buf[i])|(uint32_t(buf[i+1])<<8)|(uint32_t(buf[i+2])<<16)|(uint32_t(buf[i+3])<<24);require((x&0x7f800000)!=0x7f800000,"NaN/Inf: "+t.name);}}
    left-=n;
   }
  }
 }
#ifdef LUNA_VISUAL_CAPTURE
 captured_tensors=tensors; captured_data=data; return;
#endif
 if(patch) {patch_embed(r,data,tensors,patch);return;}
 cout<<"VALIDO: estrutura Qwen3.5-0.8B mmproj F16, 154 tensores; dados "<<(scan?"varridos: sem NaN/Inf":"nao varridos")<<".\n";
 cout<<"Isso nao autentica os pesos; confira SHA-256 de fonte confiavel. NAO executa encoder completo.\n";
}
int main(int argc,char**argv){
 bool scan=argc==3 && string(argv[2])=="--scan";
 bool patch=argc==4 && string(argv[2])=="--patch";
 if(!(argc==2||scan||patch)){cerr<<"Uso: qwen_validate ARQUIVO.gguf [--scan | --patch PATCH_F32.bin]\n";return 2;}
 try{validate(argv[1],scan,patch?argv[3]:nullptr);return 0;}catch(const exception&e){cerr<<"REJEITADO: "<<e.what()<<"\n";return 1;}
}
