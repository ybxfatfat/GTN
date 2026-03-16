mkdir model_logs
cp -rf models/DIN/logs model_logs/din_logs
cp -rf models/MPNRec/logs model_logs/mpnrec_logs
cp -rf models/NRMS/logs model_logs/nrms_logs
cp -rf models/wide_deep/logs model_logs/wide_deep_logs
tar -zcvf model_logs.tar.gz model_logs