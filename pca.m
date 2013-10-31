function [D, W_pca] = pca(X)
	Xm = bsxfun(@minus, X, mean(X));
	C = cov(Xm);
	[W_pca, D] = eig(C);
	[D, i] = sort(diag(D), 'descend');
	W_pca = W_pca(:,i);
end
