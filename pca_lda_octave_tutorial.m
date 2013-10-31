clear all; close all;

%% Load Wine Dataset

data = dlmread("wine.csv",",");
X = data(:,2:end); y = data(:,1);

%% Principal Component without Normalization

[Dpca, Wpca] = pca(X);
Xm = bsxfun(@minus, X, mean(X));
Xproj = project(Xm, Wpca(:,1:2));

cumsum(Dpca) / sum(Dpca)

disp(''); disp('Press enter to continue'); disp('');
pause

% the first component accounts for 99% of total Variance

wine1 = Xproj(find(y == 1), :);
wine2 = Xproj(find(y == 2), :);
wine3 = Xproj(find(y == 3), :);

figure;
hold on;
plot(wine1(:,1), wine1(:,2), "ro", "markersize", 10, "linewidth", 3);  
plot(wine2(:,1), wine2(:,2), "go", "markersize", 10, "linewidth", 3); 
plot(wine3(:,1), wine3(:,2), "bo", "markersize", 10, "linewidth", 3); 
title("PCA (original data)")

disp(''); disp('Press enter to continue'); disp('');
pause

%% Linear Discriminant Analysis without Normalization

[Dlda,Wlda] = lda(X,y);
Xproj = project(Xm, Wlda(:,1:2));

wine1 = Xproj(find(y == 1), :);
wine2 = Xproj(find(y == 2), :);
wine3 = Xproj(find(y == 3), :);

figure;
hold on;
plot(wine1(:,1), wine1(:,2), "ro", "markersize", 10, "linewidth", 3);  
plot(wine2(:,1), wine2(:,2), "go", "markersize", 10, "linewidth", 3); 
plot(wine3(:,1), wine3(:,2), "bo", "markersize", 10, "linewidth", 3); 
title("LDA (original data)")

disp(''); disp('Press enter to continue'); disp('');
pause

%% Fix PCA by using Normalization

Xnew = zscore(X);

[Dpca, Wpca] = pca(Xnew);
Xm = bsxfun(@minus, Xnew, mean(Xnew));
Xproj = project(Xm, Wpca(:,1:2));

cumsum(Dpca) / sum(Dpca)

disp(''); disp('Press enter to continue'); disp('');



wine1 = Xproj(find(y == 1), :);
wine2 = Xproj(find(y == 2), :);
wine3 = Xproj(find(y == 3), :);

figure;
hold on;
plot(wine1(:,1), wine1(:,2), "ro", "markersize", 10, "linewidth", 3);  
plot(wine2(:,1), wine2(:,2), "go", "markersize", 10, "linewidth", 3); 
plot(wine3(:,1), wine3(:,2), "bo", "markersize", 10, "linewidth", 3); 
title("PCA (Normalized)")

disp(''); disp('Press enter to end'); disp('');
pause


%% compare against LDA with Normalization

[Dlda,Wlda] = lda(Xnew,y);
Xproj = project(Xm, Wlda(:,1:2));

wine1 = Xproj(find(y == 1), :);
wine2 = Xproj(find(y == 2), :);
wine3 = Xproj(find(y == 3), :);

figure;
hold on;
plot(wine1(:,1), wine1(:,2), "ro", "markersize", 10, "linewidth", 3);  
plot(wine2(:,1), wine2(:,2), "go", "markersize", 10, "linewidth", 3); 
plot(wine3(:,1), wine3(:,2), "bo", "markersize", 10, "linewidth", 3); 
title("LDA (Normalized)")

disp(''); disp('Press enter to end'); disp('');
pause

