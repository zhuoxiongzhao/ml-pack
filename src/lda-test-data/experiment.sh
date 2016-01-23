#! /bin/bash

corpus="yahoo"
common_opt="-hp_opt 0 -K 3 -alpha 0.1 -beta 0.1 -total_iteration 200 -burnin_iteration 0 -log_likelihood_interval 1 ${corpus}-train"

# corpus="nips"
# common_opt="-hp_opt 0 -K 100 -alpha 0.1 -beta 0.1 -total_iteration 200 -burnin_iteration 0 -log_likelihood_interval 1 nips-train"

../lda-train $common_opt -sampler lda ${corpus}-lda 2>&1 | grep LogLike | awk -F "=" '{print $2}' | awk -F "/" '{print $1}' > ${corpus}-lda-llh
../lda-train $common_opt -sampler sparselda ${corpus}-sparselda 2>&1 | grep LogLike | awk -F "=" '{print $2}' | awk -F "/" '{print $1}' > ${corpus}-sparselda-llh

../lda-train $common_opt -sampler aliaslda -mh_step 2 ${corpus}-aliaslda-mh2 2>&1 | grep LogLike | awk -F "=" '{print $2}' | awk -F "/" '{print $1}' > ${corpus}-aliaslda-mh2-llh
../lda-train $common_opt -sampler aliaslda -mh_step 4 ${corpus}-aliaslda-mh4 2>&1 | grep LogLike | awk -F "=" '{print $2}' | awk -F "/" '{print $1}' > ${corpus}-aliaslda-mh4-llh
../lda-train $common_opt -sampler aliaslda -mh_step 8 ${corpus}-aliaslda-mh8 2>&1 | grep LogLike | awk -F "=" '{print $2}' | awk -F "/" '{print $1}' > ${corpus}-aliaslda-mh8-llh
../lda-train $common_opt -sampler aliaslda -mh_step 16 ${corpus}-aliaslda-mh16 2>&1 | grep LogLike | awk -F "=" '{print $2}' | awk -F "/" '{print $1}' > ${corpus}-aliaslda-mh16-llh

../lda-train $common_opt -sampler lightlda -mh_step 2 ${corpus}-lightlda-mh2 2>&1 | grep LogLike | awk -F "=" '{print $2}' | awk -F "/" '{print $1}' > ${corpus}-lightlda-mh2-llh
../lda-train $common_opt -sampler lightlda -mh_step 4 ${corpus}-lightlda-mh4 2>&1 | grep LogLike | awk -F "=" '{print $2}' | awk -F "/" '{print $1}' > ${corpus}-lightlda-mh4-llh
../lda-train $common_opt -sampler lightlda -mh_step 8 ${corpus}-lightlda-mh8 2>&1 | grep LogLike | awk -F "=" '{print $2}' | awk -F "/" '{print $1}' > ${corpus}-lightlda-mh8-llh
../lda-train $common_opt -sampler lightlda -mh_step 16 ${corpus}-lightlda-mh16 2>&1 | grep LogLike | awk -F "=" '{print $2}' | awk -F "/" '{print $1}' > ${corpus}-lightlda-mh16-llh

../lda-train $common_opt -sampler lightlda -mh_step 16 -enable_doc_proposal 0 ${corpus}-lightlda-mh16-word 2>&1 | grep LogLike | awk -F "=" '{print $2}' | awk -F "/" '{print $1}' > ${corpus}-lightlda-mh16-word-llh
../lda-train $common_opt -sampler lightlda -mh_step 16 -enable_word_proposal 0 ${corpus}-lightlda-mh16-doc 2>&1 | grep LogLike | awk -F "=" '{print $2}' | awk -F "/" '{print $1}' > ${corpus}-lightlda-mh16-doc-llh
