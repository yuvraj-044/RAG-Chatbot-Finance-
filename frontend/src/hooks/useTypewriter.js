import { useState, useEffect, useRef } from "react";

/**
 * Reveals text character-by-character to simulate streaming tokens.
 * @param {string} fullText - The complete text to reveal
 * @param {boolean} shouldAnimate - Whether to animate (false = show all immediately)
 * @param {number} speed - Milliseconds per character (default 12)
 * @returns {{ displayedText: string, isTyping: boolean }}
 */
export function useTypewriter(fullText, shouldAnimate = true, speed = 12) {
  const [displayedText, setDisplayedText] = useState(
    shouldAnimate ? "" : fullText
  );
  const [isTyping, setIsTyping] = useState(false);
  const indexRef = useRef(0);
  const intervalRef = useRef(null);

  useEffect(() => {
    if (!shouldAnimate || !fullText) {
      setDisplayedText(fullText || "");
      setIsTyping(false);
      return;
    }

    indexRef.current = 0;
    setDisplayedText("");
    setIsTyping(true);

    intervalRef.current = setInterval(() => {
      indexRef.current += 1;

      // Accelerate for long texts — reveal 2-3 chars at a time after 200 chars
      if (indexRef.current > 200) {
        indexRef.current += 1;
      }
      if (indexRef.current > 500) {
        indexRef.current += 2;
      }

      if (indexRef.current >= fullText.length) {
        setDisplayedText(fullText);
        setIsTyping(false);
        clearInterval(intervalRef.current);
      } else {
        setDisplayedText(fullText.slice(0, indexRef.current));
      }
    }, speed);

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [fullText, shouldAnimate, speed]);

  return { displayedText, isTyping };
}
