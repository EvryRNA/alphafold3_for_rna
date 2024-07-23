install_usalign:
	mkdir -p lib/usalign
	wget https://zhanggroup.org/US-align/bin/module/USalign.cpp -O lib/usalign/USalign.cpp
	g++ -O3 -lm -o lib/usalign/USalign lib/usalign/USalign.cpp

install_rna_assessment:
	mkdir -p lib
	git clone https://github.com/clementbernardd/RNA_assessment.git --branch scoring-version lib/rna_assessment

compute_metrics:
	python -m src.benchmark.score_computation

stats:
	python -m src.utils.stats

count_interactions:
	python -m src.utils.count_interactions

viz_alphafold:
	python -m src.viz_alphafold