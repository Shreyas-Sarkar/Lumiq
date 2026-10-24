"use client";

import { useEffect } from "react";

export default function HomeMotion() {
  useEffect(() => {
    const preference = window.matchMedia("(prefers-reduced-motion: reduce)");
    const animations = new Set<Animation>();
    const revealed = new WeakSet<Element>();
    let observer: IntersectionObserver | undefined;
    function configure() {
      observer?.disconnect();
      if (preference.matches) {
        animations.forEach((animation) => animation.cancel());
        animations.clear();
        return;
      }
      observer = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            if (!entry.isIntersecting || revealed.has(entry.target)) return;
            revealed.add(entry.target);
            observer?.unobserve(entry.target);
            const animation = entry.target.animate(
              [
                { opacity: 0, transform: "translateY(14px)" },
                { opacity: 1, transform: "translateY(0)" },
              ],
              { duration: 480, easing: "cubic-bezier(.22,1,.36,1)" },
            );
            animations.add(animation);
            animation.finished.then(
              () => animations.delete(animation),
              () => animations.delete(animation),
            );
          });
        },
        { threshold: 0.12 },
      );
      document
        .querySelectorAll(".lumiq-home [data-reveal]")
        .forEach((element) => observer?.observe(element));
    }
    configure();
    preference.addEventListener("change", configure);
    return () => {
      observer?.disconnect();
      animations.forEach((animation) => animation.cancel());
      preference.removeEventListener("change", configure);
    };
  }, []);
  return null;
}
