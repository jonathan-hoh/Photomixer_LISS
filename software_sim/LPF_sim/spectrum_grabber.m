model = 'accum_response_gold.slx';
sablock = 'accum_response_gold/Spectrum Analyzer';
cfg = get_param(sablock,'ScopeConfiguration');
cfg.CursorMeasurements.Enable = true;
cfg.ChannelMeasurements.Enable = true;
cfg.PeakFinder.Enable = true;
cfg.DistortionMeasurements.Enable = true;
sim(model)
data = getSpectrumData(cfg);
trace = data.MaxHoldTrace(1);
frequencies = data.FrequencyVector(1);
output = [frequencies, trace];
