function [kernel_c1,kernel_f1]=init_kernel(layer_c1_num,layer_f1_num)
%% 卷积核初始化
for n=1:layer_c1_num
    kernel_c1(:,:,n)=(2*rand(7,7)-ones(7,7))/12;   %5*5的-1到1的随机数除以12
end
for n=1:layer_f1_num
    kernel_f1(:,:,n)=(2*rand(11,11)-ones(11,11));   %12*12的-1到1的随机数
end
end