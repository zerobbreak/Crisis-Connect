import { useEffect } from "react";
import { useNavigate } from "react-router";

export default function RedirectHome() {
  const navigate = useNavigate();

  useEffect(() => {
    navigate("/home", { replace: true });
  }, [navigate]);

  return null;
}
