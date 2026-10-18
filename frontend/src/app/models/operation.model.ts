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
