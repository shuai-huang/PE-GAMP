addpath('./BGM') 
addpath('./main')

cluster_num = 10;
rate_num = 2;
wave_num = 6;
noise_level = 10;
image_name = 'lena';
load(strcat('./test_images_256/', image_name, '.mat'));


img_dim = 256;
im=zeros(img_dim,img_dim);

nlevel=4;

[h0, h1, f0, f1]=wfilters('db8');
[h0_db1, h1_db1, f0_db1, f1_db1]=wfilters('db1');
[h0_db2, h1_db2, f0_db2, f1_db2]=wfilters('db2');
[h0_db3, h1_db3, f0_db3, f1_db3]=wfilters('db3');
[h0_db4, h1_db4, f0_db4, f1_db4]=wfilters('db4');
[h0_db5, h1_db5, f0_db5, f1_db5]=wfilters('db5');
[h0_db6, h1_db6, f0_db6, f1_db6]=wfilters('db6');
[h0_db7, h1_db7, f0_db7, f1_db7]=wfilters('db7');

h0_sq=h0.^2;h1_sq=h1.^2;f0_sq=f0.^2;f1_sq=f1.^2;
h0_db1_sq=h0_db1.^2;h1_db1_sq=h1_db1.^2;f0_db1_sq=f0_db1.^2;f1_db1_sq=f1_db1.^2;
h0_db2_sq=h0_db2.^2;h1_db2_sq=h1_db2.^2;f0_db2_sq=f0_db2.^2;f1_db2_sq=f1_db2.^2;
h0_db3_sq=h0_db3.^2;h1_db3_sq=h1_db3.^2;f0_db3_sq=f0_db3.^2;f1_db3_sq=f1_db3.^2;
h0_db4_sq=h0_db4.^2;h1_db4_sq=h1_db4.^2;f0_db4_sq=f0_db4.^2;f1_db4_sq=f1_db4.^2;
h0_db5_sq=h0_db5.^2;h1_db5_sq=h1_db5.^2;f0_db5_sq=f0_db5.^2;f1_db5_sq=f1_db5.^2;
h0_db6_sq=h0_db6.^2;h1_db6_sq=h1_db6.^2;f0_db6_sq=f0_db6.^2;f1_db6_sq=f1_db6.^2;
h0_db7_sq=h0_db7.^2;h1_db7_sq=h1_db7.^2;f0_db7_sq=f0_db7.^2;f1_db7_sq=f1_db7.^2;

dwtmode('per');
[C,S]=wavedec2(im,nlevel,h0,h1); 
ncoef=length(C);
[C1,S1]=wavedec2(im,nlevel,h0_db1,h1_db1); 
ncoef1=length(C1);
[C2,S2]=wavedec2(im,nlevel,h0_db2,h1_db2); 
ncoef2=length(C2);
[C3,S3]=wavedec2(im,nlevel,h0_db3,h1_db3); 
ncoef3=length(C3);
[C4,S4]=wavedec2(im,nlevel,h0_db4,h1_db4); 
ncoef4=length(C4);
[C5,S5]=wavedec2(im,nlevel,h0_db5,h1_db5); 
ncoef5=length(C5);
[C6,S6]=wavedec2(im,nlevel,h0_db6,h1_db6); 
ncoef6=length(C6);
[C7,S7]=wavedec2(im,nlevel,h0_db7,h1_db7); 
ncoef7=length(C7);

switch wave_num
case 1
Psi = @(x) [wavedec2(x,nlevel,h0_db1,h1_db1)']; 
Psit = @(x) (waverec2(x(1:ncoef1),S1,f0_db1,f1_db1));

case 2
Psi = @(x) [wavedec2(x,nlevel,h0_db2,h1_db2)']; 
Psit = @(x) (waverec2(x(1:ncoef2),S2,f0_db2,f1_db2));

case 3
Psi = @(x) [wavedec2(x,nlevel,h0_db3,h1_db3)']; 
Psit = @(x) (waverec2(x(1:ncoef3),S3,f0_db3,f1_db3));

case 4
Psi = @(x) [wavedec2(x,nlevel,h0_db4,h1_db4)']; 
Psit = @(x) (waverec2(x(1:ncoef4),S4,f0_db4,f1_db4));

case 5
Psi = @(x) [wavedec2(x,nlevel,h0_db5,h1_db5)']; 
Psit = @(x) (waverec2(x(1:ncoef5),S5,f0_db5,f1_db5));

case 6
Psi = @(x) [wavedec2(x,nlevel,h0_db6,h1_db6)']; 
Psit = @(x) (waverec2(x(1:ncoef6),S6,f0_db6,f1_db6));

case 7
Psi = @(x) [wavedec2(x,nlevel,h0_db7,h1_db7)']; 
Psit = @(x) (waverec2(x(1:ncoef7),S7,f0_db7,f1_db7));

otherwise
Psi = @(x) [wavedec2(x,nlevel,h0,h1)']; 
Psit = @(x) (waverec2(x(1:ncoef),S,f0,f1));

end


switch wave_num
case 1
Psi2 = @(x) [wavedec2(x,nlevel,h0_db1_sq,h1_db1_sq)'];
Psit2 = @(x) (waverec2(x(1:ncoef1),S1,f0_db1_sq,f1_db1_sq));

case 2
Psi2 = @(x) [wavedec2(x,nlevel,h0_db2_sq,h1_db2_sq)'];
Psit2 = @(x) (waverec2(x(1:ncoef2),S2,f0_db2_sq,f1_db2_sq));

case 3
Psi2 = @(x) [wavedec2(x,nlevel,h0_db3_sq,h1_db3_sq)'];
Psit2 = @(x) (waverec2(x(1:ncoef3),S3,f0_db3_sq,f1_db3_sq));

case 4
Psi2 = @(x) [wavedec2(x,nlevel,h0_db4_sq,h1_db4_sq)'];
Psit2 = @(x) (waverec2(x(1:ncoef4),S4,f0_db4_sq,f1_db4_sq));

case 5
Psi2 = @(x) [wavedec2(x,nlevel,h0_db5_sq,h1_db5_sq)'];
Psit2 = @(x) (waverec2(x(1:ncoef5),S5,f0_db5_sq,f1_db5_sq));

case 6
Psi2 = @(x) [wavedec2(x,nlevel,h0_db6_sq,h1_db6_sq)'];
Psit2 = @(x) (waverec2(x(1:ncoef6),S6,f0_db6_sq,f1_db6_sq));

case 7
Psi2 = @(x) [wavedec2(x,nlevel,h0_db7_sq,h1_db7_sq)'];
Psit2 = @(x) (waverec2(x(1:ncoef7),S7,f0_db7_sq,f1_db7_sq));

otherwise
Psi2 = @(x) [wavedec2(x,nlevel,h0_sq,h1_sq)'];
Psit2 = @(x) (waverec2(x(1:ncoef),S,f0_sq,f1_sq));

end


img=img(1:img_dim,1:img_dim);
imSize = size(img);
rate     = rate_num/10;
N=imSize(1)*imSize(2);
M=round(rate*N);

switch wave_num
case 1
matSize = [M ncoef1];

case 2
matSize = [M ncoef2];

case 3
matSize = [M ncoef3];

case 4
matSize = [M ncoef4];

case 5
matSize = [M ncoef5];

case 6
matSize = [M ncoef6];

case 7
matSize = [M ncoef7];

otherwise
matSize = [M ncoef];

end

% generate random sampling matrix

D2=randn(M, N);
D2_norm=sqrt(sum(D2.^2));
for(j=1:N)
    D2(:,j)=D2(:,j)/D2_norm(j);
end

noise=randn(M,1);

A2 = RandCosLinTrans(D2,imSize,Psi,Psit,Psi2,Psit2,matSize);

%% noiseless measurement
Y2 = A2.mult(Psi(img));

% noisy measurement
Y2 = Y2+noise_level*noise;
Y2_l2=Psi(reshape(D2'*inv(D2*D2')*Y2,imSize));


% Use the least squre solution (l2 norm minimization) as the initialization
idx_kmeans = kmeans(Y2_l2,cluster_num,'MaxIter', 1000);
active_mean=[];
active_var=[];
active_weights=[];
for (i=1:cluster_num)
active_mean = [active_mean mean(Y2_l2(idx_kmeans==i))];
active_var = [active_var var(Y2_l2(idx_kmeans==i))];
active_weights = [active_weights length(Y2_l2(idx_kmeans==i))/length(Y2_l2)];
end
 
optPE.active_mean=active_mean';
optPE.active_var=active_var';
optPE.active_weights=active_weights';

optPE.noise_var = 1e-3;
optPE.lambda=0.5;
optPE.maxPEiter = 100;
optPE.L=cluster_num;
optGAMP.nit = 10;
[Xr, PEfin]=PEGMAMP(Y2, A2, optPE, optGAMP);

imshow(Psit(Xr), [])
psnr_val = psnr(img, Psit(Xr));

