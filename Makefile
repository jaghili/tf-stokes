train:	snapshots train.py
	python train.py

snapshots: stokessolns.edp
	FreeFem++ stokessolns.edp

clean:
	rm -rf tmp/*
	rm -rf live/js/data.js
	rm -rf trial/*.dat
