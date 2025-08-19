import React from "react";

type LogoRotationVariantProps = {
  src: string;
  alt?: string;
  width?: number;
  height?: number;
  className?: string;
};

const LogoRotationVariant = (props: LogoRotationVariantProps) => {
  const { src, alt = "", width = 40, height = 40, className = "" } = props;

  return (
    <img
      src={src}
      alt={alt}
      width={width}
      height={height}
      className={`animate-spin ${className}`}
      style={{ animationDuration: "2s" }}
    />
  );
};

export default LogoRotationVariant;