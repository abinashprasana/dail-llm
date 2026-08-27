type SealProps = {
  size?: number;
};

export function Seal({ size = 30 }: SealProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true">
      <defs>
        <linearGradient id="seal-brass" x1="16" y1="6" x2="16" y2="24" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#e2c489" />
          <stop offset="1" stopColor="#a97f3e" />
        </linearGradient>
      </defs>
      <circle cx="16" cy="16" r="14.5" fill="#07110e" stroke="url(#seal-brass)" strokeWidth="1.3" />
      <path d="M16 7.4 L23 14.1 L21.6 14.6 L16 9.9 L10.4 14.6 L9 14.1 Z" fill="url(#seal-brass)" />
      <rect x="9.9" y="14.9" width="2.15" height="6.7" rx="0.5" fill="url(#seal-brass)" />
      <rect x="14.9" y="14.9" width="2.15" height="6.7" rx="0.5" fill="url(#seal-brass)" />
      <rect x="19.9" y="14.9" width="2.15" height="6.7" rx="0.5" fill="url(#seal-brass)" />
      <rect x="8.6" y="21.9" width="14.8" height="1.5" rx="0.7" fill="url(#seal-brass)" />
    </svg>
  );
}
