import { Directive, ElementRef, Input, NgZone, OnChanges, OnDestroy, OnInit, inject } from '@angular/core';

/**
 * Mouse zoom for an image window.
 *   wheel         zoom in / out around the cursor
 *   drag          pan (when zoomed in)
 *   double-click  back to the normal view
 * Each element using the directive keeps its own zoom, so every window is independent.
 */
@Directive({ selector: '[appZoomPan]', standalone: true })
export class ZoomPanDirective implements OnInit, OnChanges, OnDestroy {
  /** Bind the image URL here: when it changes, the view resets. */
  @Input('appZoomPan') resetKey: unknown;

  private readonly host = inject<ElementRef<HTMLElement>>(ElementRef).nativeElement;
  private readonly zone = inject(NgZone);

  private readonly maxScale = 16;
  private scale = 1;
  private tx = 0;
  private ty = 0;
  private drag: { x: number; y: number; tx: number; ty: number } | null = null;

  ngOnInit(): void {
    this.zone.runOutsideAngular(() => {
      // Not passive, so the page does not scroll while zooming.
      this.host.addEventListener('wheel', this.onWheel, { passive: false });
      this.host.addEventListener('pointerdown', this.onDown);
      this.host.addEventListener('pointermove', this.onMove);
      this.host.addEventListener('pointerup', this.onUp);
      this.host.addEventListener('pointercancel', this.onUp);
      this.host.addEventListener('dblclick', this.reset);
    });
  }

  ngOnChanges(): void {
    this.reset();
  }

  ngOnDestroy(): void {
    this.host.removeEventListener('wheel', this.onWheel);
    this.host.removeEventListener('pointerdown', this.onDown);
    this.host.removeEventListener('pointermove', this.onMove);
    this.host.removeEventListener('pointerup', this.onUp);
    this.host.removeEventListener('pointercancel', this.onUp);
    this.host.removeEventListener('dblclick', this.reset);
  }

  private readonly reset = (): void => {
    this.scale = 1;
    this.tx = 0;
    this.ty = 0;
    this.apply();
  };

  private readonly onWheel = (e: WheelEvent): void => {
    if (!this.host.querySelector('img')) {
      return;
    }
    e.preventDefault();
    const rect = this.host.getBoundingClientRect();
    // Cursor position relative to the window centre (the image is centred there).
    const px = e.clientX - rect.left - rect.width / 2;
    const py = e.clientY - rect.top - rect.height / 2;

    const delta = e.deltaMode === 1 ? e.deltaY * 30 : e.deltaY;
    const next = Math.min(this.maxScale, Math.max(1, this.scale * Math.exp(-delta * 0.0015)));
    const k = next / this.scale;

    // Keep the point under the cursor fixed while the scale changes.
    this.tx = px - (px - this.tx) * k;
    this.ty = py - (py - this.ty) * k;
    this.scale = next;
    this.clamp(rect);
    this.apply();
  };

  private readonly onDown = (e: PointerEvent): void => {
    if (this.scale <= 1 || e.button !== 0) {
      return;
    }
    this.drag = { x: e.clientX, y: e.clientY, tx: this.tx, ty: this.ty };
    this.host.setPointerCapture(e.pointerId);
    this.host.classList.add('panning');
  };

  private readonly onMove = (e: PointerEvent): void => {
    if (!this.drag) {
      return;
    }
    this.tx = this.drag.tx + (e.clientX - this.drag.x);
    this.ty = this.drag.ty + (e.clientY - this.drag.y);
    this.clamp(this.host.getBoundingClientRect());
    this.apply();
  };

  private readonly onUp = (e: PointerEvent): void => {
    if (!this.drag) {
      return;
    }
    this.drag = null;
    this.host.classList.remove('panning');
    if (this.host.hasPointerCapture(e.pointerId)) {
      this.host.releasePointerCapture(e.pointerId);
    }
  };

  /** Stop the image from being dragged completely out of the window. */
  private clamp(rect: DOMRect): void {
    const maxX = (rect.width * this.scale) / 2;
    const maxY = (rect.height * this.scale) / 2;
    this.tx = Math.min(maxX, Math.max(-maxX, this.tx));
    this.ty = Math.min(maxY, Math.max(-maxY, this.ty));
  }

  private apply(): void {
    const img = this.host.querySelector('img');
    if (img) {
      img.style.transform =
        this.scale === 1 ? '' : `translate(${this.tx}px, ${this.ty}px) scale(${this.scale})`;
    }
    this.host.classList.toggle('zoomed', this.scale > 1);
  }
}
