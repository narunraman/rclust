"""Shared pytest setup. The real-cluster tests in tests/integration are opt-in (see the README,
"Testing against your own clusters"); without RCLUST_TEST_CLUSTERS they are all skipped."""

import os


def pytest_addoption(parser):
    parser.addoption(
        "--rclust-config", metavar="PATH", default=None,
        help="rclust config for the real-cluster tests (rclust's --config). Default: "
             "$RCLUST_CONFIG, else ./config.yaml, else ~/.config/rclust/config.yaml.")


def pytest_configure(config):
    # Check RCLUST_TEST_CLUSTERS up front, so a typo is one clear error rather than a collection error.
    if os.environ.get("RCLUST_TEST_CLUSTERS", "").strip():
        from integration.clusters import selection

        selection(config)


def pytest_report_header(config):
    clusters = os.environ.get("RCLUST_TEST_CLUSTERS", "").strip()
    if not clusters:
        return None
    lines = [f"rclust real-cluster tests: RCLUST_TEST_CLUSTERS={clusters} (read-only)"]
    if os.environ.get("RCLUST_TEST_SUBMIT", "").strip() == "1":
        lines.append("RCLUST_TEST_SUBMIT=1: submits ONE tiny real job (1 CPU, 1 minute, 256M) to each "
                     "selected cluster; this spends a little of your allocation.")
    return lines
