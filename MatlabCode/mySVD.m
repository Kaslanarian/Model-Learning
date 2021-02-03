function [compress_A, real_ratio] = mySVD(A, ratio)
    [U, S, V] = svd(A);
    eigs = diag(S);
    SUM = sum(eigs);
    temp = 0;

    for i = 1:length(eigs)
        temp = temp + eigs(i);

        if temp / SUM > ratio
            break
        end

    end

    real_ratio = temp / SUM
    compress_A = U(:1:i) * S(1:i, 1:i) * V(:1:i);
end
