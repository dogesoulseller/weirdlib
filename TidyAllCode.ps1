Get-ChildItem ./src,./include -Recurse -Include *.cpp,*.hpp |
	ForEach-Object {

		Write-Host ''
		Write-Host 'Now processing file' $_.Name...
		Write-Host ''

		$checks = "bugprone-*,"`
		+"modernize-*,"`
		+"performance-*,"`
		+"readability-*,-readability-magic-numbers,-readability-uppercase-literal-suffix,"`
		+"portability-*,"`
		+"clang-analyzer-*,"`
		+"misc-*,"`
		+"-clang-analyzer-osx*,-clang-analyzer-optin.osx*,-clang-analyzer-optin.mpi*,-clang-analyzer-apiModeling*"
		clang-tidy $_.FullName --checks=$checks -- -march=native -std=c++17 -Iinclude

		Write-Host ''
		Write-Host 'File done, press any key for next file...'
		$null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')
	}
