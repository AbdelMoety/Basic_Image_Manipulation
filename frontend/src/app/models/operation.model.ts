export interface CvOperation {
  id: string;
  label: string;
  group: string;
}

export interface ProcessResult {
  error?: string;
  info?: string;
  uploaded_image?: string;
  processed_image?: string;
  uploaded_image_2?: string;
}

export interface ProcessParams {
  filter_type: string;
  convert_gray: boolean;
  kernel_size: number;
  sigma: number;
  canny_sigma: number;
  thresh_val: number;
  iter_diff: number;
  local_win: number;
  local_c: number;
  noise_mean: number;
  noise_sigma: number;
  sp_amount: number;
  sp_salt_ratio: number;
  uniform_low: number;
  uniform_high: number;
  cutoff_low: number;
  cutoff_high: number;
  alpha: number;
  beta: number;
}

/** The two output windows. */
export type WindowKey = 'out1' | 'out2';

/** One finished run, kept in the session history. */
export interface RunEntry {
  id: string;
  title: string;
  opId: string;
  /** Path/URL returned by the backend (pass through mediaUrl() before use). */
  url: string;
  /** Charts (histograms) can be viewed but not fed back in as an input image. */
  isChart: boolean;
}

/** What an output window is currently showing. */
export interface WindowView {
  url: string | null;
  label: string;
  empty: string;
  run: RunEntry | null;
}
