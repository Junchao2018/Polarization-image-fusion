function [ img_out ] = GrayStretch( img_in )
% m = mean(img_in(:));
% s = std(img_in(:));
% img_in(img_in>m+3*s) = m+3*s;
% img_in(img_in<m-3*s) = m-3*s;
% img_out = (img_in-m+3*s)*255/(6*s);
maxv = max(img_in(:));
minv = min(img_in(:));
img_out = (img_in-minv)*255/(maxv-minv);
%%
% meanv = mean(img_out(:));
% stdv = std(img_out(:));
% img_out = (img_out-meanv)/(stdv+eps);
