function [D, W_lda] = lda(X,y)
	% n = dimensions
	[temp n] = size(X);
	labels = unique(y); 
	C = length(labels);
	Sw = zeros(n,n);
	Sb = zeros(n,n);
	mu = mean(X);
	
	for i = 1:C
		Xi = X(find(y == labels(i)),:);
		[m temp] = size(Xi);
		mu_i = mean(Xi);
		XMi = bsxfun(@minus, Xi, mu_i);
		Sw = Sw + ( XMi' * XMi);
		MiM = mu_i - mu;
		Sb = Sb + m * MiM' * MiM;
	end
	
	[W_lda, D] = eig(Sw\Sb);
	[D, i] = sort(diag(D), 'descend');
	W_lda = W_lda(:,i);
end
