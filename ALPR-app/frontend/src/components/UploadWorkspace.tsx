import type { ChangeEvent } from 'react';
import styles from '../App.module.css';
import type { PipelineStatus } from '../types/alpr';

interface UploadWorkspaceProps {
  status: PipelineStatus;
  selectedFileName: string | null;
  onFileChange: (event: ChangeEvent<HTMLInputElement>) => void;
  onStart: () => void;
  errorMsg: string | null;
}

export function UploadWorkspace({
  status,
  selectedFileName,
  onFileChange,
  onStart,
  errorMsg,
}: UploadWorkspaceProps) {
  const busy = status === 'PENDING' || status === 'UPLOADING';

  return (
    <>
      {/* Drop zone */}
      <div className={styles.dropZone}>
        <input
          type="file"
          accept="image/jpeg,image/png,image/jpg"
          onChange={onFileChange}
          aria-label="Upload vehicle image"
        />
        <div className={styles.dropIcon}>📷</div>
        <p className={styles.dropTitle}>
          {selectedFileName ? selectedFileName : 'Drop vehicle image here'}
        </p>
        <p className={styles.dropSub}>
          {selectedFileName
            ? 'Click to change file'
            : 'or click to browse — JPEG / PNG supported'}
        </p>
        {selectedFileName && (
          <span className={styles.fileSelectedBadge}>
            ✓ Ready to scan
          </span>
        )}
      </div>

      {/* Run button */}
      <button
        className={styles.runButton}
        onClick={onStart}
        disabled={!selectedFileName || busy}
      >
        {status === 'UPLOADING'
          ? '⟳ Uploading…'
          : status === 'PENDING'
          ? '⟳ Running Pipeline…'
          : '⚡ Run ALPR Pipeline'}
      </button>

      {/* Processing state */}
      {status === 'PENDING' && (
        <>
          <div className={styles.processingCard}>
            <div className={styles.scanWrapper}>
              <div className={styles.scanLine} />
            </div>
            <div className={styles.processingText}>
              <h4>Pipeline Running</h4>
              <p>
                Classifying environment, applying restoration, running YOLO detection
                and TrOCR — this takes a few seconds.
              </p>
            </div>
          </div>
          
          <div className={styles.rollingLoaderContainer}>
            <div className={styles.rollingLoader} />
            <p className={styles.rollingLoaderText}>Just a minute...</p>
          </div>
        </>
      )}

      {/* Error */}
      {errorMsg && (
        <div className={styles.errorAlert}>
          <span>⚠</span> {errorMsg}
        </div>
      )}
    </>
  );
}
