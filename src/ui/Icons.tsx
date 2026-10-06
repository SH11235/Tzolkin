import type { CSSProperties, ReactNode } from 'react';

export type IconName =
  | 'corn'
  | 'wood'
  | 'stone'
  | 'gold'
  | 'skull'
  | 'worker'
  | 'sun'
  | 'temple'
  | 'gear'
  | 'arrow'
  | 'book'
  | 'undo'
  | 'save';

export function Icon({
  name,
  size = 20,
  className = '',
}: {
  name: IconName;
  size?: number;
  className?: string;
}) {
  const paths: Record<IconName, ReactNode> = {
    corn: (
      <>
        <path d="M9 17C4 13 4 5 10 3c5 2 5 9 0 14Z" />
        <path d="M7 7h6M7 10h6M8 13h4M10 3v14M10 21c-6-1-8-5-8-9 5 1 7 4 8 9Zm0 0c6-1 8-5 8-9-5 1-7 4-8 9Z" />
      </>
    ),
    wood: (
      <>
        <path d="m12 2 6 7h-3l5 7h-6v6h-4v-6H4l5-7H6Z" />
        <path d="M12 8v8" />
      </>
    ),
    stone: (
      <>
        <path d="m3 17 3-9 8-4 6 6 1 8-10 3Z" />
        <path d="m6 8 5 5 9-3M11 13v8" />
      </>
    ),
    gold: (
      <>
        <path d="m5 8 14 0 3 11H2Z" />
        <path d="m5 8 3-4h8l3 4M5 8l3 11m11-11-3 11" />
      </>
    ),
    skull: (
      <>
        <path d="M5 15c-5-12 19-16 14 0l-3 2v4H8v-4Z" />
        <circle cx="8" cy="11" r="2" />
        <circle cx="16" cy="11" r="2" />
        <path d="m10 16 2-3 2 3M11 18v3m3-3v3" />
      </>
    ),
    worker: (
      <>
        <circle cx="12" cy="5" r="3" />
        <path d="M8 10h8l5 7h-5v5h-3v-5h-2v5H8v-5H3Z" />
      </>
    ),
    sun: (
      <>
        <circle cx="12" cy="12" r="5" />
        <path d="M12 1v3m0 16v3M1 12h3m16 0h3M4 4l2 2m12 12 2 2M4 20l2-2M18 6l2-2" />
      </>
    ),
    temple: (
      <>
        <path d="M3 21h18v-4H3Zm3-4h12v-4H6Zm3-4h6V9H9Zm2-4h2V4h-2ZM1 21h22" />
      </>
    ),
    gear: (
      <>
        <path d="m9 2 6 0 1 4 4 1 2 5-3 3 0 4-5 3-3-3-4 0-3-5 3-3 0-4Z" />
        <circle cx="12" cy="12" r="4" />
      </>
    ),
    arrow: (
      <>
        <path d="M4 12h15m-6-6 6 6-6 6" />
      </>
    ),
    book: (
      <>
        <path d="M12 5c-3-3-7-3-10-2v16c4-1 7 0 10 2 3-2 6-3 10-2V3c-3-1-7-1-10 2Zm0 0v16" />
      </>
    ),
    undo: (
      <>
        <path d="m8 3-5 5 5 5M3 8h10c10 0 10 13 0 13h-3" />
      </>
    ),
    save: (
      <>
        <path d="M4 3h13l4 4v14H3V3Zm3 0v6h9V3M7 21v-8h10v8" />
      </>
    ),
  };
  return (
    <svg
      className={`icon ${className}`}
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinejoin="round"
      strokeLinecap="round"
      aria-hidden="true"
    >
      {paths[name]}
    </svg>
  );
}

export function CalendarArt({ round = 1, className = '' }: { round?: number; className?: string }) {
  return (
    <div className={`calendar-art ${className}`} aria-hidden="true">
      <svg
        viewBox="0 0 400 400"
        className="calendar-teeth"
        style={{ '--angle': `${((round - 1) * 360) / 26}deg` } as CSSProperties}
      >
        <g fill="currentColor">
          {Array.from({ length: 26 }, (_, i) => (
            <rect
              key={i}
              x="187"
              y="2"
              width="26"
              height="37"
              rx="3"
              transform={`rotate(${(i * 360) / 26} 200 200)`}
            />
          ))}
        </g>
        <circle cx="200" cy="200" r="177" fill="currentColor" />
        <circle cx="200" cy="200" r="165" className="calendar-inset" />
        <circle cx="200" cy="200" r="144" fill="none" stroke="currentColor" strokeWidth="1" />
        {Array.from({ length: 26 }, (_, i) => (
          <g key={i} transform={`rotate(${(i * 360) / 26} 200 200)`}>
            <path d="M196 42h8v14h-8Z" fill="none" stroke="currentColor" strokeWidth="2" />
            <path d="M200 65v14" stroke="currentColor" />
          </g>
        ))}
        <circle cx="200" cy="200" r="113" fill="none" stroke="currentColor" strokeWidth="3" />
        <circle cx="200" cy="200" r="106" fill="none" stroke="currentColor" strokeWidth="1" />
      </svg>
      <svg viewBox="0 0 160 160" className="calendar-face">
        <g fill="none" stroke="currentColor" strokeWidth="3">
          <path d="M40 44V24h20V14h40v10h20v20M31 49h98v20l-10 10v26l-18 22H59l-18-22V79L31 69Z" />
          <path d="M42 57h28v15H42Zm48 0h28v15H90ZM70 75h20l9 19H61ZM55 105h50v13H55M67 105v13m13-13v13m13-13v13M20 48H8v49h12m120-49h12v49h-12M50 137h60M60 145h40" />
          <path d="M50 38h60M62 30h36" />
        </g>
      </svg>
    </div>
  );
}
