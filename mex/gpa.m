%function [Xnew,mu] = gpa(X)
% gpa Performs a (full) generalized Procrustes analysis in 2D   
%   [Xnew ,mu] = gpa(X) creates a matrix M-by-N Matrix of aligned shapes
%   from an M-by-N matrix X. M is the number of shapes and N/2 is the
%   number of landmarks. 1:N/2 are the first N/2 coordinates and N/2+1:N
%   are the second N/2 coordinates of the N/2 landmarks for each shape. 
%   Xnew are the aligned shapes.
%   mu are the eigenvector representing the most variance, becoming the
%   mean of the distributions of each landmark.
%   
%   Auther Poul K. Sørensen, 3th March of 2013
%
clear all;
mex -I'C:\Development\CIL\external\eigen-eigen-5097c01bcdc4\' -I'..\modules\core\include\' -I'..\modules\algorithms\include\' gpa.cpp

close all
load shapes;
shapes = reshape(shapes, size(shapes,1)*size(shapes, 2), size(shapes,3))';
ashapes = ashapes';

n=240;
figure(1)
subplot(1,3,1)
plot(shapes(1:n,2:2:end),-shapes(1:n,1:2:end),'.')
subplot(1,3,2)
plot(ashapes(1:n,2:2:end),-ashapes(1:n,1:2:end),'.'); hold on;
plot(mu(2,:),-mu(1,:),'ow');
subplot(1,3,3)
[myashapes meanshape] = gpa([shapes(:,1:2:end) shapes(:,2:2:end)]);
%axis equal square
plot(myashapes(1:n,end/2+1:end),-myashapes(1:n,1:end/2),'.'); hold on;
plot(meanshape(end/2+1:end),-meanshape(1:end/2),'ow');

X = [myashapes(1:n,end/2+1:end) -myashapes(1:n,1:end/2)];
Y = [ashapes(1:n,2:2:end),-ashapes(1:n,1:2:end)];
distance = sum(sum((X-Y).^2))./numel(X) + sum(([ meanshape(end/2+1:end)' -meanshape(1:end/2)']-[ mu(2,:) -mu(1,:)]).^2)./numel(mu)