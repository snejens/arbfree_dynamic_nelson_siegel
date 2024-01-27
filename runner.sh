#!/bin/bash

cd ~/PycharmProjects/arbfree-dyn-ns && . ~/Nextcloud/Documents/GBS/Thesis/venv/bin/activate

#nice -n 19 python -m arbfree_dyn_ns.main &
#nice -n 19 python -m arbfree_dyn_ns.main --cutoff_min "2020-01-01" &
#nice -n 19 python -m arbfree_dyn_ns.main --cutoff_min "2021-01-01" &
#nice -n 19 python -m arbfree_dyn_ns.main --cutoff_min "2015-01-01" &
#nice -n 19 python -m arbfree_dyn_ns.main --test
#nice -n 19 python -m arbfree_dyn_ns.main --cutoff_max "2020-01-01" &
#nice -n 19 python -m arbfree_dyn_ns.main --cutoff_min "2015-01-01" --cutoff_max "2020-01-01" &
nice -n 19 python -m arbfree_dyn_ns.main --n 5 --cutoff_max "2010-01-01" &
nice -n 19 python -m arbfree_dyn_ns.main --n 5 --cutoff_max "2009-01-01" &
#nice -n 19 python -m arbfree_dyn_ns.main --cutoff_max "2007-01-01" &
