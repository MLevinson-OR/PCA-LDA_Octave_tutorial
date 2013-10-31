function Z = zscore(X)
	Z = bsxfun(@rdivide, bsxfun(@minus, X, mean(X)), std(X));
end

