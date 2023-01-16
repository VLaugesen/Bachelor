I = imread('s2707_e01.tif');
J = I(6000+(0:4*1024-1),6000+(0:4*1024-1));
K = imgaussfilt(J,3)>230;
imshowpair(J,K)