function n = GetRandPhase(ResetFlag)
	if (nargin < 1);ResetFlag=0;end
	if (ResetFlag)
		rng(0, 'simdTwister');
	end
	n = randi([1 3]);
end
