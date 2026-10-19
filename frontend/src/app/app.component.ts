import { CommonModule } from '@angular/common';
import { Component, ElementRef, OnInit, ViewChild, inject } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ZoomPanDirective } from './directives/zoom-pan.directive';

import {
  CvOperation,
  ProcessParams,
  RunEntry,
  WindowKey,
  WindowView,
} from './models/operation.model';
import { VisionApiService } from './services/vision-api.service';

const AUTO_GRAY_OPS = new Set([
  'rgb2gray',
  'threshold',
  'iter_threshold',
  'local_threshold',
  'equalize',
  'auto_canny',
]);

const MAX_RUNS = 30;

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule, ZoomPanDirective],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss',
})
export class AppComponent implements OnInit {
  private readonly api = inject(VisionApiService);

  @ViewChild('primaryInput') private primaryInput?: ElementRef<HTMLInputElement>;

  title = 'Digital Image Processing Studio';
  operations: CvOperation[] = [];
  groupedOperations: { group: string; items: CvOperation[] }[] = [];

  selectedOperation = 'sobel';
  primaryFile: File | null = null;
  secondaryFile: File | null = null;
  primaryPreview: string | null = null;
  secondaryPreview: string | null = null;

  params: ProcessParams = {
    filter_type: 'sobel',
    convert_gray: false,
    kernel_size: 3,
    sigma: 1.0,
    canny_low: 50,
    canny_high: 150,
    thresh_val: 128,
    iter_diff: 0.001,
    local_win: 15,
    local_c: 5.0,
    noise_mean: 0,
    noise_sigma: 25,
    sp_amount: 0.05,
    sp_salt_ratio: 0.5,
    uniform_low: -25,
    uniform_high: 25,
    cutoff_low: 25,
    cutoff_high: 25,
    alpha: 0.5,
    beta: 0.5,
  };

  loading = false;
  furthering = false;
  error: string | null = null;
  info: string | null = null;

  /** Finished runs, newest first. */
  runs: RunEntry[] = [];
  private runCount = 0;
  /** How many times "Use as input" has been applied (0 = untouched upload). */
  chainStep = 0;

  // ---- windows ----
  /** Which input image the input window shows (2 is only used by hybrid). */
  inputTab: 1 | 2 = 1;
  /** What each output window shows: a run id, or '' for empty. */
  slots: Record<WindowKey, string> = { out1: '', out2: '' };
  /** Which output window the next run will appear in (chosen in Controls). */
  target: WindowKey = 'out1';
  readonly outputKeys: WindowKey[] = ['out1', 'out2'];

  ngOnInit(): void {
    this.api.listOperations().subscribe({
      next: (res) => {
        this.operations = res.operations;
        this.groupedOperations = this.groupByCategory(res.operations);
      },
      error: () => {
        this.error = 'Backend not reachable. Start it with: python manage.py runserver';
      },
    });
  }

  // ---- which parameter fields to show ----
  get op(): string {
    return this.selectedOperation;
  }
  get showGrayToggle(): boolean {
    return !AUTO_GRAY_OPS.has(this.op);
  }
  get showImage2(): boolean {
    return this.op === 'hybrid';
  }
  get showKernel(): boolean {
    return ['mean', 'gaussian', 'median', 'auto_canny'].includes(this.op);
  }
  get showSigma(): boolean {
    return this.op === 'gaussian';
  }
  get showCannyThresh(): boolean {
    return this.op === 'auto_canny';
  }
  get showThresh(): boolean {
    return this.op === 'threshold';
  }
  get showIterDiff(): boolean {
    return this.op === 'iter_threshold';
  }
  get showLocal(): boolean {
    return this.op === 'local_threshold';
  }
  get showNoise(): boolean {
    return this.op === 'awgn';
  }
  get showSaltPepper(): boolean {
    return this.op === 'salt_pepper';
  }
  get showUniform(): boolean {
    return this.op === 'uniform_noise';
  }
  get showCutoffLow(): boolean {
    return this.op === 'lowpass' || this.op === 'hybrid';
  }
  get showCutoffHigh(): boolean {
    return this.op === 'highpass' || this.op === 'hybrid';
  }
  get showHybridWeights(): boolean {
    return this.op === 'hybrid';
  }
  get hasParams(): boolean {
    return (
      this.showKernel || this.showSigma || this.showCannyThresh || this.showThresh ||
      this.showIterDiff || this.showLocal || this.showNoise || this.showSaltPepper ||
      this.showUniform || this.showCutoffLow || this.showCutoffHigh || this.showHybridWeights ||
      this.showGrayToggle
    );
  }

  // ---- windows ----
  get inputUrl(): string | null {
    return this.inputTab === 2 ? this.secondaryPreview : this.primaryPreview;
  }

  view(key: WindowKey): WindowView {
    const id = this.slots[key];
    const run = id ? (this.runs.find((r) => r.id === id) ?? null) : null;
    const name = key === 'out1' ? 'Output 1' : 'Output 2';
    return {
      url: run ? this.mediaUrl(run.url) : null,
      label: run ? run.title : 'Nothing here yet',
      empty: id
        ? 'That run is no longer in the list.'
        : `Choose “${name}” under “Show result in”, then press Process image.`,
      run,
    };
  }

  setSlot(key: WindowKey, value: string): void {
    this.slots[key] = value;
  }

  canFurther(run: RunEntry | null): boolean {
    return !!run && !run.isChart && !this.loading && !this.furthering;
  }

  /** Make a result the input image, so another operation can be applied to it. */
  async further(run: RunEntry | null): Promise<void> {
    const url = run ? this.mediaUrl(run.url) : null;
    if (!run || !url || !this.canFurther(run)) {
      return;
    }
    this.furthering = true;
    this.error = null;
    try {
      const blob = await this.fetchBlob(url);
      const ext = (url.split('?')[0].match(/\.(png|jpe?g|bmp|webp|tiff?)$/i)?.[1] ?? 'png').toLowerCase();
      // Short client-side name: the backend prefixes names on every pass, so reusing
      // the server filename would eventually hit the filesystem limit.
      this.chainStep += 1;
      const file = new File([blob], `further_${this.chainStep}.${ext}`, {
        type: blob.type || `image/${ext === 'jpg' ? 'jpeg' : ext}`,
      });
      this.setPrimary(file);
      this.info = `${run.title} is now the input image. Pick an operation and press Process.`;
    } catch {
      this.error = 'Could not load that result to use it as the input.';
    } finally {
      this.furthering = false;
    }
  }

  async download(key: WindowKey): Promise<void> {
    const v = this.view(key);
    if (!v.url) {
      return;
    }
    try {
      const blob = await this.fetchBlob(v.url);
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `${v.label.replace(/[^\w]+/g, '_').replace(/^_|_$/g, '') || 'image'}.png`;
      a.click();
      URL.revokeObjectURL(a.href);
    } catch {
      this.error = 'Download failed.';
    }
  }

  // ---- inputs ----
  onOperationChange(): void {
    this.params.filter_type = this.selectedOperation;
    if (!this.showImage2) {
      this.inputTab = 1;
    }
    this.error = null;
    this.info = null;
  }

  onPrimarySelected(event: Event): void {
    const file = (event.target as HTMLInputElement).files?.[0] ?? null;
    this.chainStep = 0;
    this.setPrimary(file, false);
  }

  onSecondarySelected(event: Event): void {
    const file = (event.target as HTMLInputElement).files?.[0] ?? null;
    this.secondaryFile = file;
    this.revokePreview(this.secondaryPreview);
    this.secondaryPreview = file ? URL.createObjectURL(file) : null;
  }

  process(): void {
    if (!this.primaryFile) {
      this.error = 'Upload Image 1 first.';
      return;
    }
    if (this.op === 'hybrid' && !this.secondaryFile) {
      this.error = 'Hybrid mode needs a second image.';
      return;
    }

    const form = new FormData();
    form.append('image', this.primaryFile);
    if (this.secondaryFile) {
      form.append('image_2', this.secondaryFile);
    }
    form.append('filter_type', this.selectedOperation);
    if (this.params.convert_gray) {
      form.append('convert_gray', 'on');
    }

    const numericKeys: (keyof ProcessParams)[] = [
      'kernel_size', 'sigma', 'canny_low', 'canny_high', 'thresh_val', 'iter_diff', 'local_win', 'local_c',
      'noise_mean', 'noise_sigma', 'sp_amount', 'sp_salt_ratio', 'uniform_low', 'uniform_high',
      'cutoff_low', 'cutoff_high', 'alpha', 'beta',
    ];
    for (const key of numericKeys) {
      form.append(key, String(this.params[key]));
    }

    this.loading = true;
    this.error = null;
    this.info = null;
    const opId = this.selectedOperation;

    this.api.processImage(form).subscribe({
      next: (res) => {
        this.loading = false;
        if (res.error) {
          this.error = res.error;
          return;
        }
        if (!res.processed_image) {
          this.error = 'The backend returned no output image.';
          return;
        }
        this.runCount += 1;
        const label = this.operations.find((o) => o.id === opId)?.label ?? opId;
        const id = `r${this.runCount}`;
        this.slots[this.target] = id;
        this.runs = [
          {
            id,
            title: `#${this.runCount} ${label}`,
            opId,
            url: res.processed_image,
            isChart: opId === 'plot_histograms',
          },
          ...this.runs,
        ].slice(0, MAX_RUNS);
        this.info = res.info ?? null;
      },
      error: (err) => {
        this.loading = false;
        this.error = err.error?.error ?? 'Processing failed. Check the backend logs.';
      },
    });
  }

  mediaUrl(path: string | undefined): string | null {
    return this.api.mediaUrl(path);
  }

  // ---- helpers ----
  private setPrimary(file: File | null, syncInput = true): void {
    this.primaryFile = file;
    this.revokePreview(this.primaryPreview);
    this.primaryPreview = file ? URL.createObjectURL(file) : null;
    if (file && syncInput) {
      this.syncPrimaryInput(file);
    }
  }

  private async fetchBlob(url: string): Promise<Blob> {
    const resp = await fetch(url);
    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}`);
    }
    return resp.blob();
  }

  private groupByCategory(ops: CvOperation[]): { group: string; items: CvOperation[] }[] {
    const map = new Map<string, CvOperation[]>();
    for (const op of ops) {
      const list = map.get(op.group) ?? [];
      list.push(op);
      map.set(op.group, list);
    }
    return Array.from(map.entries()).map(([group, items]) => ({ group, items }));
  }

  /** Make the file input display the promoted file instead of the old upload. */
  private syncPrimaryInput(file: File): void {
    const input = this.primaryInput?.nativeElement;
    if (!input) {
      return;
    }
    try {
      const dt = new DataTransfer();
      dt.items.add(file);
      input.files = dt.files;
    } catch {
      /* Older browsers: the preview and primaryFile are still correct. */
    }
  }

  private revokePreview(url: string | null): void {
    if (url) {
      URL.revokeObjectURL(url);
    }
  }
}
