clear all;
clc;
opts.kernel_size=15;
s=2;
ret = sqrt(0.5);
maxitr=max(floor(log(5/min(opts.kernel_size))/log(ret)),0);
retv=ret.^[0:maxitr];
k1list=ceil(opts.kernel_size*retv);
k1list=k1list+(mod(k1list,2)==0);   %%全部转化为奇数
k2list=ceil(opts.kernel_size*retv);
k2list=k2list+(mod(k2list,2)==0);
ks=blurCore(4);
ks1 = resizeKer(ks,1/ret,k1list(s),k2list(s));