% Input image
I = imread('/Users/jrh630/Downloads/1-70.tiff');
J = mean(I,3);
figure(1); imagesc(J); colormap(gray); colorbar
figure(2); histogram(J(:), 100);

% Threshold
K = J > 200;
figure(3); imagesc(K); colormap(gray); colorbar

% Analyse components
CC = bwconncomp(K);
idx = CC.PixelIdxList{3};
L = zeros(size(K));
L(idx) = 1;
figure(4); imagesc(L); colormap(gray); colorbar