const nextJest = require("next/jest");

const createJestConfig = nextJest({ dir: "./" });

// MSW's interceptors depend on a few packages that ship ESM only. next/jest
// already lets a few packages through its node_modules transform-ignore list
// (next.config.js transpilePackages); add rettime to that same allowlist here
// instead of to transpilePackages, so the production build is untouched.
const ESM_ONLY_TEST_DEPS = ["rettime", "until-async", "@open-draft/deferred-promise"];

function allowEsmOnly(config) {
  return {
    ...config,
    transformIgnorePatterns: config.transformIgnorePatterns.map((pattern) =>
      pattern.replace("(?!(", `(?!(${ESM_ONLY_TEST_DEPS.join("|")}|`),
    ),
  };
}

module.exports = async () => ({
  projects: [
    await createJestConfig({
      displayName: "node",
      testEnvironment: "node",
      testMatch: [
        "<rootDir>/__tests__/utils/**/*.test.ts",
        "<rootDir>/__tests__/domain/**/*.test.ts",
      ],
    })(),
    allowEsmOnly(
      await createJestConfig({
        displayName: "jsdom",
        // jsdom with Node's fetch stack, so MSW can intercept requests.
        testEnvironment: "<rootDir>/src/test/jest-environment.js",
        testMatch: [
          "<rootDir>/__tests__/components/**/*.test.tsx",
          "<rootDir>/__tests__/pages/**/*.test.tsx",
          "<rootDir>/__tests__/lib/**/*.test.ts",
          "<rootDir>/__tests__/lib/**/*.test.tsx",
        ],
        setupFilesAfterEnv: ["<rootDir>/jest.setup.js"],
      })(),
    ),
  ],
});
