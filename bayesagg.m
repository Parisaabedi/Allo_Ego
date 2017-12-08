function sigma = bayesagg(lambda,ccm12,cm)
    mucm = lambda * cm;
    d = abs(cm - mucm);
    r = size(ccm12,1);
    ccm12 = ccm12 + eye(r) .* d;    
    sigma =  (lambda * ccm12 * lambda');
end